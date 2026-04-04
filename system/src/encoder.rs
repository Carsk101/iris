use std::io::Write;
use std::process::{Command, Stdio};
use anyhow::{Result, Context, bail};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Encoder { Av1Nvenc, HevcNvenc, LibX264 }

impl Encoder {
    pub fn container(self) -> &'static str { "matroska" }
    pub fn to_u8(self)  -> u8 { match self { Encoder::Av1Nvenc=>0, Encoder::HevcNvenc=>1, Encoder::LibX264=>2 } }
    pub fn from_u8(v: u8) -> Self { match v { 0=>Encoder::Av1Nvenc, 1=>Encoder::HevcNvenc, _=>Encoder::LibX264 } }
    pub fn label(self) -> &'static str { match self { Encoder::Av1Nvenc=>"av1_nvenc", Encoder::HevcNvenc=>"hevc_nvenc", Encoder::LibX264=>"libx264" } }
}

/// Runtime probe — actually test-encode a frame. libsvtav1 -crf 0 is NOT lossless.
/// Probe order: av1_nvenc → hevc_nvenc → libx264 -qp 0 (guaranteed lossless).
pub fn detect() -> Encoder {
    if probe_nvenc("av1_nvenc",  &["-tune","lossless","-rc","constqp","-qp","0"]) {
        eprintln!("[iris] encoder: av1_nvenc (RTX 4000+)"); return Encoder::Av1Nvenc;
    }
    if probe_nvenc("hevc_nvenc", &["-tune","lossless","-rc","constqp","-qp","0"]) {
        eprintln!("[iris] encoder: hevc_nvenc (RTX 3000)"); return Encoder::HevcNvenc;
    }
    eprintln!("[iris] encoder: libx264 -qp 0 (CPU lossless fallback)");
    Encoder::LibX264
}

fn probe_nvenc(codec: &str, extra: &[&str]) -> bool {
    let frame = vec![128u8; 64 * 64 * 3 / 2];
    let mut args: Vec<String> = vec![
        "-hide_banner".into(),"-loglevel".into(),"error".into(),
        "-f".into(),"rawvideo".into(),"-pix_fmt".into(),"yuv420p".into(),
        "-s".into(),"64x64".into(),"-r".into(),"1".into(),
        "-i".into(),"pipe:0".into(),"-c:v".into(),codec.into(),
    ];
    for a in extra { args.push((*a).into()); }
    args.extend(["-frames:v".into(),"1".into(),"-f".into(),"null".into(),"-".into()]);
    let mut child = match Command::new("ffmpeg").args(&args)
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
        Ok(c)=>c, Err(_)=>return false,
    };
    if let Some(s) = child.stdin.as_mut() { let _ = s.write_all(&frame); }
    child.wait().map(|s| s.success()).unwrap_or(false)
}

pub fn encode(yuv: &[u8], w: usize, h: usize, frames: usize, enc: Encoder) -> Result<Vec<u8>> {
    let sz   = format!("{}x{}", w, h);
    let mut args: Vec<String> = vec![
        "-hide_banner".into(),"-loglevel".into(),"error".into(),
        "-f".into(),"rawvideo".into(),"-pix_fmt".into(),"yuv420p".into(),
        "-s".into(),sz,"-r".into(),"1".into(),"-i".into(),"pipe:0".into(),
    ];
    match enc {
        Encoder::Av1Nvenc => args.extend([
            "-c:v".into(),"av1_nvenc".into(),"-preset".into(),"p7".into(),
            "-tune".into(),"lossless".into(),"-rc".into(),"constqp".into(),
            "-qp".into(),"0".into(),"-multipass".into(),"fullres".into(),
            "-bf".into(),"3".into(),"-refs".into(),"16".into(),
        ]),
        Encoder::HevcNvenc => args.extend([
            "-c:v".into(),"hevc_nvenc".into(),"-preset".into(),"p7".into(),
            "-tune".into(),"lossless".into(),"-rc".into(),"constqp".into(),
            "-qp".into(),"0".into(),"-bf".into(),"3".into(),"-refs".into(),"16".into(),
        ]),
        Encoder::LibX264 => args.extend([
            "-c:v".into(),"libx264".into(),"-qp".into(),"0".into(),
            "-preset".into(),"slow".into(),  // slow = better ratio at lossless QP
        ]),
    }
    args.extend(["-f".into(),"matroska".into(),"pipe:1".into()]);

    let mut child = Command::new("ffmpeg").args(&args)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
        .spawn().context("failed to spawn ffmpeg")?;
    child.stdin.as_mut().context("stdin")?.write_all(yuv)?;
    let out = child.wait_with_output()?;
    if !out.status.success() { bail!("ffmpeg encode failed: {}", out.status); }
    eprintln!("[iris] encoded {}×{} ×{} → {} bytes ({})", w, h, frames, out.stdout.len(), enc.label());
    Ok(out.stdout)
}

pub fn decode(video: &[u8], w: usize, h: usize, frames: usize, enc: Encoder) -> Result<Vec<u8>> {
    let sz = format!("{}x{}", w, h);
    let args: Vec<String> = vec![
        "-hide_banner".into(),"-loglevel".into(),"error".into(),
        "-f".into(),"matroska".into(),"-i".into(),"pipe:0".into(),
        "-f".into(),"rawvideo".into(),"-pix_fmt".into(),"yuv420p".into(),
        "-s".into(),sz,"pipe:1".into(),
    ];
    let mut child = Command::new("ffmpeg").args(&args)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
        .spawn().context("failed to spawn ffmpeg decode")?;
    child.stdin.as_mut().context("stdin")?.write_all(video)?;
    let out = child.wait_with_output()?;
    if !out.status.success() { bail!("ffmpeg decode failed: {}", out.status); }
    eprintln!("[iris] decoded {}×{} ×{} → {} bytes", w, h, frames, out.stdout.len());
    Ok(out.stdout)
}
