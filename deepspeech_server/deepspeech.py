#TODO ADD PATH PARA O ARQUIVO CSV DE SENTENÇAS AO JSON
#TODO CONECTAR A TRANSCRICAO COM O CLASSIFICAR E O RESULTADO

import io
import logging
from collections import namedtuple
import scipy.io.wavfile as wav

import rx
from cyclotron import Component
from cyclotron_std.logging import Log
from deepspeech import Model

# Classify module
from deepspeech_server.Classify import Classify

BERTIMBAU_MODEL = 'neuralmind/bert-base-portuguese-cased'
THRESHOLD = 0.95    # Similaridade mínima aceita entre a frase transcrita e a do classificador (correlação mínima)
#FILE_CSV = '/home/igoxy/Documentos/deepspeech-server/sentences.csv'

#sts = None  # Classify

Sink = namedtuple('Sink', ['speech'])
Source = namedtuple('Source', ['text', 'log'])

# Sink events
Scorer = namedtuple('Scorer', ['scorer', 'lm_alpha', 'lm_beta'])
Scorer.__new__.__defaults__ = (None, None, None)

Initialize = namedtuple('Initialize', ['model', 'scorer', 'beam_width', 'csv'])
Initialize.__new__.__defaults__ = (None,)

SpeechToText = namedtuple('SpeechToText', ['data', 'context'])

# Sourc eevents
TextResult = namedtuple('TextResult', ['text', 'context'])
TextError = namedtuple('TextError', ['error', 'context'])


def make_driver(loop=None):
    def driver(sink):
        model = None
        log_observer = None

        sts = None  # Classify

        def on_log_subscribe(observer, scheduler):
            nonlocal log_observer
            log_observer = observer

        def log(message, level=logging.DEBUG):
            if log_observer is not None:
                log_observer.on_next(Log(
                    logger=__name__,
                    level=level,
                    message=message,
                ))

        def setup_model(model_path, scorer, beam_width):
            log("creating model {} with scorer {}...".format(model_path, scorer))
            model = Model(model_path)

            if scorer.scorer is not None:
                model.enableExternalScorer(scorer.scorer)
                if scorer.lm_alpha is not None and scorer.lm_beta is not None:
                    if model.setScorerAlphaBeta(scorer.lm_alpha, scorer.lm_beta) != 0:
                        raise RuntimeError("Unable to set scorer parameters")

            if beam_width is not None:
                if model.setBeamWidth(beam_width) != 0:
                    raise RuntimeError("Unable to set beam width")

            log("model is ready.")
            return model

        #Inicializar o objeto para o classificador
        def setup_classify(csv):
            log(f'Iniciando o classificador com o modelo {BERTIMBAU_MODEL} e sentenças do arquivo {csv}')
            return Classify(BERTIMBAU_MODEL, csv)


        def subscribe(observer, scheduler):
            def on_deepspeech_request(item):
                nonlocal model
                nonlocal sts

                if type(item) is SpeechToText:
                    if model is not None:
                        try:
                            _, audio = wav.read(io.BytesIO(item.data))
                            # convert to mono.
                            # todo: move to a component or just a function here
                            if len(audio.shape) > 1:
                                audio = audio[:, 0]
                            text = model.stt(audio)
                            log("STT result: {}".format(text))
                            
                            # -- Aplicação do classificador após o modelo deepspeech --
                            result_sts = sts.find_more_similar(text)    # Obtém o mais similar do conjunto de frases
                            log(f'From STS result: {result_sts[0]} \nScore: {result_sts[1]}')

                            if result_sts[1] >= THRESHOLD:  # Se tiver uma alta correlação, substitui a transcrição original
                                text = result_sts[0]
                            
                            # -- Aplicação do classificador após o modelo deepspeech --
                                
                            observer.on_next(rx.just(TextResult(
                                text=text,
                                context=item.context,
                            )))
                        except Exception as e:
                            log("STT error: {}".format(e), level=logging.ERROR)
                            observer.on_next(rx.throw(TextError(
                                error=e,
                                context=item.context,
                            )))
                elif type(item) is Initialize:
                    log("initialize: {}".format(item))
                    model = setup_model(
                        item.model, item.scorer, item.beam_width)
                    
                    sts = setup_classify(item.csv) # Inicializa o objeto do classificador
                
                else:
                    log("unknown item: {}".format(item), level=logging.CRITICAL)
                    observer.on_error(
                        "Unknown item type: {}".format(type(item)))

            sink.speech.subscribe(lambda item: on_deepspeech_request(item))

        return Source(
            text=rx.create(subscribe),
            log=rx.create(on_log_subscribe),
        )

    return Component(call=driver, input=Sink)
