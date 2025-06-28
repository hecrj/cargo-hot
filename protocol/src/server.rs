use crate::Result;

use subsecond::JumpTable;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net;

pub struct Server {
    socket: net::TcpListener,
}

pub struct Connection {
    socket: net::TcpStream,
    aslr_reference: usize,
}

impl Server {
    pub async fn bind() -> Result<Self> {
        let server = net::TcpListener::bind("127.0.0.1:1100").await?;

        Ok(Self { socket: server })
    }

    pub async fn accept(&mut self) -> Result<Connection> {
        let (mut client, _) = self.socket.accept().await?;
        client.readable().await?;

        let mut buffer = [0; std::mem::size_of::<usize>()];
        client.read_exact(&mut buffer).await?;

        Ok(Connection {
            socket: client,
            aslr_reference: usize::from_be_bytes(buffer),
        })
    }
}

impl Connection {
    pub fn aslr_reference(&self) -> usize {
        self.aslr_reference
    }

    pub async fn patch(&mut self, patch: &JumpTable) -> Result<()> {
        let bytes = bincode::serde::encode_to_vec(patch, bincode::config::standard())?;

        self.socket.writable().await?;
        self.socket.write_all(&bytes.len().to_be_bytes()).await?;
        self.socket.write_all(&bytes).await?;
        self.socket.flush().await?;
        Ok(())
    }
}
