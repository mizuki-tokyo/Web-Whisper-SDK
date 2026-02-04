import asyncio
import logging
import whisper
from threading import Thread
import queue
import logging

logger = logging.getLogger(__name__)

############################################################
# WhisperProcessor class
############################################################
class WhisperProcessor:

    _whisper_model = None

    @classmethod
    def is_model_loaded(cls):
        return cls._whisper_model is not None

    def __init__(self):
        self._recognition_queue = queue.Queue()
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self._language = "en"

    @classmethod
    def create(cls):
        return WhisperProcessor() if cls._whisper_model else None

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        """ When set to None, the language is automatically detected.
        However, Whisper estimates the language from the first 30 seconds of speech
        and determines the subsequent recognition language according to that estimate,
        so short speech will result in poor accuracy. """

        # if not value:
        #     raise ValueError("Language value is empty")
        if value == "":
            value = None # auto detection
        self._language = value

    def _worker(self):
        """音声認識のワーカースレッド"""
        while True:
            try:
                task = self._recognition_queue.get(timeout=1)
                if task is None:  # 終了シグナル
                    break
                
                audio_data, session_id, speech_id, callback, running_loop, prompt  = task
                result = self._recognize_audio(audio_data, prompt)

                # 結果をメインループで実行
                if callback and running_loop:
                    # メインループにタスクをスケジュール
                    future = asyncio.run_coroutine_threadsafe(
                        callback(result, session_id, speech_id), 
                        running_loop
                    )
                    # 結果を待機（オプション）
                    try:
                        future.result(timeout=5.0)  # 5秒でタイムアウト
                    except Exception as e:
                        logger.error(f"Failed to send recognition result: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Recognition worker error: {e}")
                
    def _recognize_audio(self, audio_data, prompt = None):
        """音声認識を実行"""
        try:
            if not WhisperProcessor._whisper_model:
                return {"error": "Whisper model not available"}
            
            # 音声データを Whisper で認識
            result = WhisperProcessor._whisper_model.transcribe(
                audio_data,
                language=self._language,
                initial_prompt=prompt,
                fp16=False,
                #fp16=True,
                verbose=False,
                # VAD で既に発話区間を特定済みなので無音判定は緩く
                no_speech_threshold=0.05,  # 大幅に下げる（デフォルト: 0.6）
                # 前文脈は利用しない（セグメントが独立しているため）
                condition_on_previous_text=False,
                # 圧縮率チェックは通常通り
                compression_ratio_threshold=2.4,
                # 信頼度は標準的に
                #logprob_threshold=-1.0,
                # 探索は標準的に
                patience=1.0,
                beam_size=5,
                # 温度は決定的に
                temperature=0.0
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Speech recognigtion result: {result} (lang:{self._language} prompt:{prompt}) ")
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    }
                    for seg in result["segments"]
                ]
            }
        
        except Exception as e:
            logger.error(f"Whisper recognition error: {e}")
            return {"error": str(e)}
        
    def recognize_async(self, audio_data, session_id, speech_id, callback, running_loop, prompt=None):
        """非同期音声認識をキューに追加"""
        try:
            self._recognition_queue.put((audio_data, session_id, speech_id, callback, running_loop, prompt))
            logger.info(f"Audio recognition queued for session {session_id} (speech: {speech_id})")
        except Exception as e:
            logger.error(f"Failed to queue recognition: {e}")



############################################################
# Whisper モデルをロード
############################################################
try:
    # 軽量なbaseモデルを使用（必要に応じてlarge-v3などに変更可能）
    ############################################################3
    # Model list
    ############################################################3
    # tiny（多言語）
    # tiny.en（英語専用）
    # base（多言語）
    # base.en（英語専用）
    # small（多言語）
    # small.en（英語専用）
    # medium（多言語）
    # medium.en（英語専用）
    # large（多言語）
    # large-v2（多言語、性能強化版）
    # large-v3（多言語、最新強化版：Mel128 & Cantonese トークン対応）
    ############################################################3

    import os
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    if not WHISPER_MODEL:
        WHISPER_MODEL = "base"

    # ~/.cache/whisper
    # import torch
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps" # 2025/7/7 現在、M1/M2/M3 Mac には未対応
    # WhisperProcessor._whisper_model = whisper.load_model(WHISPER_MODEL, device=device)

    WhisperProcessor._whisper_model = whisper.load_model(WHISPER_MODEL)
    logger.info(f"Whisper model '{WHISPER_MODEL}' loaded successfully (model info:{WhisperProcessor._whisper_model.dims})")

except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    WhisperProcessor._whisper_model = None
