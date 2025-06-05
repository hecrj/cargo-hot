# cargo-hot

A very experimental attempt at implementing the [`subsecond`] server protocol as a standalone `cargo` command.

Currently just an exploration. Very broken! Will eat your laundry!

# Credits

All credits go to [Dioxus] for figuring out an amazing cross-platform way to hotpatch Rust binaries.
Most of the code is taken from their [`dioxus-cli`] tool.

[Dioxus]: https://github.com/DioxusLabs/dioxus
[`dioxus-cli`]: https://github.com/DioxusLabs/dioxus/tree/main/packages/cli
[`subsecond`]: https://docs.rs/subsecond/0.7.0-alpha.1/subsecond/index.html
