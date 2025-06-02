use std::thread;
use std::time::Duration;

fn main() {
    cargo_hot_protocol::inject();

    loop {
        subsecond::call(|| {
            println!("Hello, world!");
        });

        thread::sleep(Duration::from_secs(1));
    }
}
