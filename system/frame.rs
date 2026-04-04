/// Frame layout engine for iris
///
/// Maps sorted bytes into grayscale video frames optimized for
/// AV1/HEVC inter-frame prediction. The core insight: sorted bytes
/// have maximum run-length structure. When laid sequentially across
/// frames, NVENC's motion estimator finds those runs as temporally
/// stable blocks and compresses them aggressively.
///
/// Layout: row-major, left-to-right, top-to-bottom, frame-by-frame.
/// Each pixel = one byte of sorted data (grayscale luma, Y plane only).

pub const FRAME_WIDTH: usize = 1920;
pub const FRAME_HEIGHT: usize = 1080;
pub const FRAME_PIXELS: usize = FRAME_WIDTH * FRAME_HEIGHT;

pub struct FrameLayout {
    pub width: usize,
    pub height: usize,
    pub frame_count: usize,
    pub padded_len: usize,
    pub original_len: usize,
}

impl FrameLayout {
    pub fn from_data_len(len: usize) -> Self {
        let (width, height) = choose_dimensions(len);
        let pixels_per_frame = width * height;
        let frame_count = (len + pixels_per_frame - 1) / pixels_per_frame;
        let padded_len = frame_count * pixels_per_frame;
        FrameLayout { width, height, frame_count, padded_len, original_len: len }
    }
}

fn choose_dimensions(len: usize) -> (usize, usize) {
    if len <= 320 * 240 { return (320, 240); }
    if len <= 854 * 480 { return (854, 480); }
    if len <= 1280 * 720 { return (1280, 720); }
    (FRAME_WIDTH, FRAME_HEIGHT)
}

pub fn pack_frames(sorted_data: &[u8], layout: &FrameLayout) -> Vec<u8> {
    let pixels_per_frame = layout.width * layout.height;
    let uv_size = pixels_per_frame / 4;
    let frame_size = pixels_per_frame + uv_size + uv_size;
    let total_size = layout.frame_count * frame_size;

    let mut frames = vec![0u8; total_size];

    for frame_idx in 0..layout.frame_count {
        let frame_start = frame_idx * frame_size;
        let y_start = frame_start;
        let cb_start = frame_start + pixels_per_frame;
        let cr_start = cb_start + uv_size;

        let data_offset = frame_idx * pixels_per_frame;
        for pixel in 0..pixels_per_frame {
            let data_idx = data_offset + pixel;
            frames[y_start + pixel] = if data_idx < sorted_data.len() {
                sorted_data[data_idx]
            } else {
                0x00
            };
        }

        for i in 0..uv_size {
            frames[cb_start + i] = 0x80;
            frames[cr_start + i] = 0x80;
        }
    }

    frames
}

pub fn unpack_frames(frame_data: &[u8], layout: &FrameLayout) -> Vec<u8> {
    let pixels_per_frame = layout.width * layout.height;
    let uv_size = pixels_per_frame / 4;
    let frame_size = pixels_per_frame + uv_size + uv_size;

    let mut out = Vec::with_capacity(layout.original_len);

    for frame_idx in 0..layout.frame_count {
        let y_start = frame_idx * frame_size;
        let data_offset = frame_idx * pixels_per_frame;

        for pixel in 0..pixels_per_frame {
            let data_idx = data_offset + pixel;
            if data_idx < layout.original_len {
                out.push(frame_data[y_start + pixel]);
            }
        }
    }

    out
}
