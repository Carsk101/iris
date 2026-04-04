use std::io::Write;
use std::process::{Command, Stdio};
use anyhow::{Result, Context, bail};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Encoder {
    Av1Nvenc,
    HevcNvenc,
    LibSvtAv1,
}

impl Encoder {
    pub fn name(&self) -> &'static str {
        match self {
            Encoder::Av1Nvenc  => "av1_nvenc",
            Encoder::HevcNvenc => "hevc_nvenc",
            Encoder::LibSvtAv1 => "libsvtav1",
        }
    }

    pub fn container(&self) -> &'static str {
        match self {
            Encoder::LibSvtAv1 => "ivf",
            _ => "matroska",
        }
    }
}

pub fn detect_encoder() -> Encoder {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output();

    let encoders = match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(_) => return Encoder::LibSvtAv1,
    };

    if encoders.contains("av1_nvenc") {
        eprintln!("[iris] encoder: av1_nvenc (RTX 4000+ detected)");
        Encoder::Av1Nvenc
    } else if encoders.contains("hevc_nvenc") {
        eprintln!("[iris] encoder: hevc_nvenc (RTX 3000 fallback)");
        Encoder::HevcNvenc
    } else {
        eprintln!("[iris] encoder: libsvtav1 (CPU fallback — no NVENC found)");
        Encoder::LibSvtAv1
    }
}

pub fn encode(
    yuv_frames: &[u8],
    width: usize,
    height: usize,
    frame_count: usize,
    encoder: Encoder,
) -> Result<Vec<u8>> {
    let size_str = format!("{}x{}", width, height);
    let container = encoder.container();

    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-f".into(), "rawvideo".into(),
        "-pix_fmt".into(), "yuv420p".into(),
        "-s".into(), size_str,
        "-r".into(), "1".into(),
        "-i".into(), "pipe:0".into(),
    ];

    match encoder {
        Encoder::Av1Nvenc => {
            args.extend([
                "-c:v".into(), "av1_nvenc".into(),
                "-preset".into(), "p7".into(),
                "-tune".into(), "lossless".into(),
                "-rc".into(), "constqp".into(),
                "-qp".into(), "0".into(),
                "-multipass".into(), "fullres".into(),
                "-bf".into(), "3".into(),
                "-refs".into(), "16".into(),
            ]);
        }
        Encoder::HevcNvenc => {
            args.extend([
                "-c:v".into(), "hevc_nvenc".into(),
                "-preset".into(), "p7".into(),
                "-tune".into(), "lossless".into(),
                "-rc".into(), "constqp".into(),
                "-qp".into(), "0".into(),
                "-bf".into(), "3".into(),
                "-refs".into(), "16".into(),
            ]);
        }
        Encoder::LibSvtAv1 => {
            args.extend([
                "-c:v".into(), "libsvtav1".into(),
                "-crf".into(), "0".into(),
                "-preset".into(), "8".into(),
                "-g".into(), "240".into(),
            ]);
        }
    }

    args.extend(["-f".into(), container.into(), "pipe:1".into()]);

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn ffmpeg — is it installed and in PATH?")?;

    {
        let stdin = child.stdin.as_mut().context("failed to open ffmpeg stdin")?;
        stdin.write_all(yuv_frames).context("failed to write frames to ffmpeg")?;
    }

    let output = child.wait_with_output().context("ffmpeg encode failed")?;
    if !output.status.success() {
        bail!("ffmpeg encoder exited with status: {}", output.status);
    }

    eprintln!(
        "[iris] encoded {} frames ({}x{}) -> {} bytes",
        frame_count, width, height, output.stdout.len()
    );

    Ok(output.stdout)
}

pub fn decode(
    video_data: &[u8],
    width: usize,
    height: usize,
    frame_count: usize,
    encoder: Encoder,
) -> Result<Vec<u8>> {
    let size_str = format!("{}x{}", width, height);
    let container = encoder.container();

    let args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-f".into(), container.into(),
        "-i".into(), "pipe:0".into(),
        "-f".into(), "rawvideo".into(),
        "-pix_fmt".into(), "yuv420p".into(),
        "-s".into(), size_str,
        "pipe:1".into(),
    ];

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn ffmpeg for decode")?;

    {
        let stdin = child.stdin.as_mut().context("failed to open ffmpeg stdin")?;
        stdin.write_all(video_data).context("failed to write video to ffmpeg")?;
    }

    let output = child.wait_with_output().context("ffmpeg decode failed")?;
    if !output.status.success() {
        bail!("ffmpeg decoder exited with status: {}", output.status);
    }

    let expected = width * height * frame_count * 3 / 2;
    eprintln!(
        "[iris] decoded {} bytes -> {} frames ({} expected bytes, got {})",
        video_data.len(), frame_count, expected, output.stdout.len()
    );

    Ok(output.stdout)
}
