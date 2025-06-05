use cargo_hot_protocol::subsecond;

use std::thread;
use std::time::Duration;

fn main() {
    cargo_hot_protocol::connect();

    loop {
        subsecond::call(|| {
            println!("Hello, world!");
        });

        thread::sleep(Duration::from_secs(1));
    }
}
