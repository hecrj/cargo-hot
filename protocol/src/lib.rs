pub use subsecond;

#[cfg(feature = "server")]
pub mod server;

pub type Result<T> = anyhow::Result<T>;

use subsecond::JumpTable;

use std::io::{Read, Write};
use std::net;
use std::sync::Once;
use std::thread;
use std::time::Duration;

static CLIENT: Once = Once::new();

pub fn connect() {
    CLIENT.call_once(|| {
        let aslr_reference = subsecond::aslr_reference();

        // TODO: Wasm support
        let _ = thread::spawn(move || {
            loop {
                if let Err(error) = run(aslr_reference) {
                    log::trace!("connection lost: {error}");
                }

                thread::sleep(Duration::from_secs(5));
            }
        });
    });
}

fn run(aslr_reference: usize) -> Result<()> {
    let mut server = net::TcpStream::connect("127.0.0.1:1100")?;
    log::info!("Connected to `cargo-hot`");

    server.write_all(&usize::to_be_bytes(aslr_reference))?;
    server.flush()?;

    let mut size = [0; std::mem::size_of::<usize>()];
    let mut buffer = Vec::new();

    loop {
        server.read_exact(&mut size)?;

        let start = std::time::Instant::now();
        let n = usize::from_be_bytes(size);

        buffer.resize(n, 0);
        server.read_exact(&mut buffer[..n])?;

        log::trace!("Read patch with {n} bytes in {:?}", start.elapsed());

        let (patch, _): (JumpTable, _) =
            bincode::serde::decode_from_slice(&buffer[..n], bincode::config::standard())?;
        log::trace!("Decoded jumptable in {:?}", start.elapsed());

        let entries = patch.map.len();

        #[allow(unsafe_code)]
        unsafe {
            subsecond::apply_patch(patch)?;
        }

        log::info!(
            "Hotpatch applied ({entries} entries) in {:?}",
            start.elapsed()
        );
    }
}
