use log::warn;
use soloud::{audio::Wav, AudioExt, LoadExt, Soloud};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

const COIN_SOUND: &[u8] = include_bytes!("../assets/coin-sound.mp3");

pub struct BlockSoundPlayer {
    sender: SyncSender<()>,
}

impl BlockSoundPlayer {
    pub fn new() -> Option<Self> {
        let (sender, receiver) = sync_channel(4);
        if let Err(error) = thread::Builder::new().name("keryx-block-sound".to_string()).spawn(move || run(receiver)) {
            warn!("Block celebration sound unavailable: {error}");
            return None;
        }
        Some(Self { sender })
    }

    pub fn play(&self) {
        let _ = self.sender.try_send(());
    }
}

fn run(receiver: Receiver<()>) {
    // Keep audio initialization and decoding off the mining and UI threads.
    if receiver.recv().is_err() {
        return;
    }

    let engine = match Soloud::default() {
        Ok(engine) => engine,
        Err(error) => {
            warn!("Block celebration sound unavailable: {error}");
            return;
        }
    };
    let mut sound = Wav::default();
    if let Err(error) = sound.load_mem(COIN_SOUND) {
        warn!("Block celebration sound could not be decoded: {error}");
        return;
    }

    engine.play(&sound);
    while receiver.recv().is_ok() {
        engine.play(&sound);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn embedded_coin_sound_matches_reviewed_asset() {
        assert_eq!(
            format!("{:x}", Sha256::digest(COIN_SOUND)),
            "440976a31e0ce6f391edc1fac122efde73232c5f07e2c0c984e24230d7dbc087"
        );
    }

    #[test]
    fn embedded_coin_sound_decodes() {
        let mut sound = Wav::default();
        assert!(sound.load_mem(COIN_SOUND).is_ok());
    }
}
