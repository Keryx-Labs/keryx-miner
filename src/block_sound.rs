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
        if let Err(error) = thread::Builder::new()
            .name("keryx-block-sound".to_string())
            .spawn(move || run(receiver))
        {
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
    // Initialize audio lazily so rigs that never find a block do not open an output device.
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
    use super::COIN_SOUND;

    #[test]
    fn embedded_coin_sound_is_present() {
        assert!(COIN_SOUND.len() > 10_000);
    }
}
