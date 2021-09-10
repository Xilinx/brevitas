import argparse
import copy
import os

from ruamel.yaml import YAML
import random
import torch

from brevitas_examples.speech_to_text.quartznet import AudioToTextDataLayer
from brevitas_examples.speech_to_text.quartznet.helpers import post_process_predictions
from brevitas_examples.speech_to_text.quartznet.helpers import post_process_transcripts
from brevitas_examples.speech_to_text.quartznet.helpers import word_error_rate
import torch.backends.cudnn as cudnn
from brevitas_examples.speech_to_text.quartznet import model_with_cfg

SEED = 123456



parser = argparse.ArgumentParser(description='Quartznet')
parser.add_argument("--data-json", type=str, required=True)
parser.add_argument("--gpu", type=int, required=False, help='GPU number')
parser.add_argument('--pretrained', action='store_true', help='Load pretrained checkpoint')
parser.add_argument('--model', type=str, default='quant_quartznet_perchannelscaling_4b', help='Name of the model')
parser.add_argument('--batch-size', default=64, type=int, help='Batch size')


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    args = parser.parse_args()

    model, cfg = model_with_cfg(args.model, args.pretrained, export_mode=False)
    topology_file = cfg.get('MODEL', 'TOPOLOGY_FILE')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    topology_path = os.path.join(current_dir, 'cfg', 'topology', topology_file)
    yaml = YAML(typ="safe")
    with open(topology_path) as f:
        quartnzet_params = yaml.load(f)

    vocab = quartnzet_params['labels']
    sample_rate = quartnzet_params['sample_rate']

    eval_dl_params = copy.deepcopy(quartnzet_params["AudioToTextDataLayer"])
    eval_dl_params.update(quartnzet_params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    data_layer = AudioToTextDataLayer(
        manifest_filepath=args.data_json,
        sample_rate=sample_rate,
        labels=vocab,
        batch_size=args.batch_size,
        **eval_dl_params)

    N = len(data_layer)
    print('Evaluating {0} examples'.format(N))

    # Set Eval mode
    model.eval()

    encoder_weights = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_weights = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    print('================================')
    print(
        f"Number of parameters in encoder: {encoder_weights}")
    print(
        f"Number of parameters in decoder: {decoder_weights}")
    print(
        f"Total number of parameters in decoder: "
        f"{encoder_weights + decoder_weights}")
    print('================================')


    if args.gpu is not None:
        cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)
        loc = 'cuda:{}'.format(args.gpu)
    else:
        loc = 'cpu'
    print(f'Running on device: {loc}')
    model.to(loc)
    
    predictions = []
    transcripts = []
    transcripts_len = []
    with torch.no_grad():
        for data in data_layer.data_iterator:
            tensors = []
            for d in data:
                tensors.append(d.to(loc))

            audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = tensors
            predictions_e1 = model(tensors)
            predictions.append(predictions_e1)
            transcripts.append(transcript_e1)
            transcripts_len.append(transcript_len_e1)

        greedy_hypotheses = post_process_predictions(predictions, vocab)
        references = post_process_transcripts(transcripts, transcripts_len, vocab)
        wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
        print("Greedy WER {:.2f}%".format(wer*100))


if __name__ == '__main__':
    main()
