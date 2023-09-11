import logging
from typing import Dict, Iterable, List, Optional

from patronum.processing import IProcessor
from patronum.processing.tool import convert_features_to_dataset
from patronum.processing.sample import Sample, SampleBasket


logger = logging.getLogger(__name__)


class ICLSProcessor(IProcessor):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.csv",
        dev_filename=None,
        test_filename="test.csv",
        dev_split=0.1,
        dev_stratification=False,
        delimiter="\t",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        multilabel=False,
        header=0,
        proxies=None,
        max_samples=None,
        text_column_name="text",
        **kwargs,
    ):
        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples
        self.dev_stratification = dev_stratification

        logger.debug("Currently no support in Processor for returning problematic ids")

        super(ICLSProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )

        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(
                name="text_classification",
                metric=metric,
                label_list=label_list,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
                task_type=task_type,
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def file_to_dicts(self, file: str) -> List[Dict]:
        raise NotImplementedError

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: Optional[List[int]] = None, return_baskets: bool = False, debug: bool = False
    ):
        if indices is None:
            indices = []
        baskets = []
        # Tokenize in batches
        texts = [x["text"] for x in dicts]
        tokenized_batch = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )

        input_ids_batch = tokenized_batch["input_ids"]
        segment_ids_batch = tokenized_batch["token_type_ids"]
        padding_masks_batch = tokenized_batch["attention_mask"]
        tokens_batch = [x.tokens for x in tokenized_batch.encodings]

        for dictionary, input_ids, segment_ids, padding_mask, tokens in zip(
            dicts, input_ids_batch, segment_ids_batch, padding_masks_batch, tokens_batch
        ):
            tokenized = {}
            if debug:
                tokenized["tokens"] = tokens

            feat_dict = {"input_ids": input_ids, "padding_mask": padding_mask, "segment_ids": segment_ids}

            # Create labels
            # i.e. not inference
            if not return_baskets:
                label_dict = self.convert_labels(dictionary)
                feat_dict.update(label_dict)

            # Add Basket to baskets
            curr_sample = Sample(id="", clear_text=dictionary, tokenized=tokenized, features=[feat_dict])
            curr_basket = SampleBasket(id_internal=None, raw=dictionary, id_external=None, samples=[curr_sample])
            baskets.append(curr_basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(n_samples=1, baskets=baskets)

        # TODO populate problematic ids
        problematic_ids: set = set()
        dataset, tensornames = self._create_dataset(baskets)
        if return_baskets:
            return dataset, tensornames, problematic_ids, baskets
        else:
            return dataset, tensornames, problematic_ids

    def _create_dataset(self, baskets: List[SampleBasket]):
        features_flat: List = []
        basket_to_remove = []
        for basket in baskets:
            if self._check_sample_features(basket):
                if not isinstance(basket.samples, Iterable):
                    raise ValueError("basket.samples must contain a list of samples.")
                for sample in basket.samples:
                    if sample.features is None:
                        raise ValueError("sample.features must not be None.")
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names


__all__ = ["ICLSProcessor"]
