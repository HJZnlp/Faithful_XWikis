# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    remove=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        # print()
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            # print("aa")
            # print(prev_output_tokens)
            # exit()
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = src_lengths.sum().item()

    fewins = torch.BoolTensor([s['few'] for s in samples ])

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'few': fewins,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        # print(prev_output_tokens)
        # print(samples)
        # exit()

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights


    stop_list=[1, 2, 3, 7, 9, 12, 13, 14, 20, 29, 33, 35, 41, 44, 51, 55, 67, 80, 88, 95, 96, 97, 101, 107, 108, 110, 133, 139, 160, 170, 183, 218, 234, 288, 344, 387, 395, 439, 441, 447, 453, 506, 601, 618, 639, 642, \
    675, 704, 753, 756, 761, 762, 805, 828, 900, 932, 956, 1218, 1254, 1281, 1283, 1292, 1376, 1553, 1657, 1669, 1807, 1833, 1899, 1916, 2171, 2298, 2360, 2364, 2409, 2443, 2496, 2681, 2682, 2747, 2803, 2806, 2816,\
     2853, 2950, 3057, 3126, 3226, 3498, 3539, 3639, 3685, 3786, 3868, 3887, 3931, 4046, 4549, 4731, 5033, 5605, 5698, 5770, 5789, 6041, 6094, 6634, 6774, 6860, 7065, 7437, 7562, 8032, 8107, 8302, 8379, 9999, 10843,\
      12485, 12635, 12957, 13435, 13645, 14599, 15041, 15397, 17718, 19438, 20268, 20591, 24142, 24186, 31946, 35061, 35296, 35975, 36914, 53330, 61258, 66567, 68031, 136562, 140576, 164052]
    # generate masks for target summaries
    # single sequence with the same length of target
    if samples[0].get('noise_labels', None) is not None:


        # TODO
        ### prepare tensor as needed to match target sequence tokens
        # tgt_sz = batch['target'].shape
        # print(batch['target'].shape)
        batch['noise_labels'] = torch.ones(batch['target'].shape,dtype=torch.int)
        batch['noise_labels'].requires_grad = False

        # samples= samples.index_select(0, sort_order)

        for idx, (tgt, new_order) in enumerate(zip(batch['target'],sort_order)):
            label=samples[new_order]["noise_labels"]
            # print(label)
            if 0 in label.data:

                # id 2 is the end token of one sentence
                # find all ends of sentence in one summary
                sen_separator=(tgt == 2).nonzero(as_tuple=True)[0]

                # find 0 in labels
                mask_sentence_ids=(label == 0).nonzero(as_tuple=True)[0]

                # label=torch.ones(len(tgt),dtype=torch.int)

                for mask_id in mask_sentence_ids:
                    # if the first sentence should be masked
                    if mask_id==0:
                        # if the first sentence need masked
                        # mask from index 1 (index 0 should not be masked)
                        # old version
                        # batch['noise_labels'][idx,1:sen_separator[mask_id]+1]=0

                        # new version
                        if remove:
                            for iii in range(1, sen_separator[mask_id] + 1):
                                if batch['target'][idx, iii] not in stop_list:
                                    batch['noise_labels'][idx, iii] = 0
                        # old version
                        else:
                            batch['noise_labels'][idx,1:sen_separator[mask_id]+1]=0
                    else:
                        #  mask from the end of last sentence +1 to the end token of current sentence (id=2)
                        # batch['noise_labels'][idx,sen_separator[mask_id-1]+1:sen_separator[mask_id]+1]=0

                        # new version
                        if remove:
                            for iii in range(sen_separator[mask_id-1]+1,sen_separator[mask_id]+1):
                                if batch['target'][idx, iii] not in stop_list:
                                    batch['noise_labels'][idx, iii] = 0
                        else:
                            # old version
                            batch['noise_labels'][idx,sen_separator[mask_id-1]+1:sen_separator[mask_id]+1]=0


                    # try:
                    #     if mask_id==0:
                    #         # if the first sentence need masked
                    #         # mask from index 1 (index 0 should not be masked)
                    #         batch['noise_labels'][idx,1:sen_separator[mask_id]+1]=0
                    #     else:
                    #         #  mask from the end of last sentence +1 to the end token of current sentence (id=2)
                    #         batch['noise_labels'][idx,sen_separator[mask_id-1]+1:sen_separator[mask_id]+1]=0
                    # except:

                    #     print(batch['target'])
                    #     print(label)
                    #     for sam in samples:
                    #         print(sam["noise_labels"])
                    #     print(mask_sentence_ids)
                    #     print(sen_separator)
                    #     # print(batch['target'].shape)
                    #     print(mask_id)
                    #     print(idx)
                    #     print(batch['target'].shape)
                    #     print(sort_order)
                    #     exit()

            # print(batch['target'][idx])
            # print(batch['noise_labels'][idx])
            # exit()
            # else:
                # label.data=torch.ones(len(tgt),dtype=torch.int)


        # batch['net_input']['noise_labels'] = batch['noise_labels']
        # print(batch['noise_labels'])

        # batch["target"]
        # shape (2,100)
        # shape of samples (2,)

        # print('Shape of target tensor?', tgt_sz)

        # print('noise tensor?', len(samples))
        # noise_labels = []
        ## prepare tensor, take care of order of elements in the batch

        # batch['noise_labels'] = noise_labels
        # exit()

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        noise_dataset=None,
        append_bos=False, eos=None,
        num_buckets=0,
        remove=True,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples {} == {}"\
                .format(len(src), len(tgt))
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        self.noise_dataset = noise_dataset
        self.remove=remove
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset
            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info('bucketing source lengths: {}'.format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info('bucketing target lengths: {}'.format(list(self.tgt.buckets)))

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens)
                for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        few = False
        invert_op = getattr(self.tgt, "isFewInstance", None)
        if callable(invert_op):
            few = self.tgt.isFewInstance(index)
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'few': few,
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        if self.noise_dataset is not None:
            example['noise_labels'] = self.noise_dataset[index]

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            remove=self.remove
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
