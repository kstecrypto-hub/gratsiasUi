class AudioError(Exception):
    category = "audio"


class InvalidAudioError(AudioError):
    category = "invalid_audio"


class AudioToolError(AudioError):
    category = "audio_tool"


class AudioSegmentationCancelledError(AudioError):
    category = "cancelled"
