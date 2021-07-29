import os
import numpy as np
import tensorflow as tf
import pickle
import copy

from physion.data.data_utils import Filter


class SequenceNewDataProvider(object):
    '''
    Sequence data provider, outputs a sequence of data of the requested length.
    This data provider supports data filtering 
    This data provider uses new dataset interface in tensorflow
    '''
    def __init__(
            self,
            data,
            enqueue_batch_size,
            sources,
            sequence_len,
            buffer_size,
            main_source_key='full_particles',
            test=False,
            delta_time=1,
            filter_rule=None,
            use_legacy_sequence_mode=False,
            file_pattern='*.tfrecords',
            seed=None,
            shuffle=True,
            map_pcall_num=48,
            subsources=[],
            shift_selector = None,
            repeat=True,
            debug=False,
            *args,
            **kwargs):

        self.data = data
        self.enqueue_batch_size = enqueue_batch_size
        self.sources = sources
        self.sequence_len = sequence_len
        self.delta_time = delta_time
        self.filter_rule = filter_rule
        self.use_legacy_sequence_mode = use_legacy_sequence_mode
        self.map_pcall_num = map_pcall_num
        self.test = test
        self.repeat = repeat
        self.shuffle = shuffle
        self.file_pattern = file_pattern
        self.seed = seed
        self.buffer_size = buffer_size
        self.all_sources = copy.deepcopy(sources)
        self.subsources = subsources
        self.main_source_key = main_source_key
        self.shift_selector = shift_selector
        self.debug = debug

        if self.test:
            if self.shift_selector is None:
                self.shift_selector = slice(1, 2)
            self.repeat = False
            self.shuffle = False
            self.key_completion = False
        else:
            self.key_completion = True

        if self.debug:
            self.shuffle = False
            self.seed = 0

        assert self.delta_time >= 1, \
                ('delta time has to be at least 1')
        assert self.sequence_len >= 1, \
                ('sequence length has to be at least 1')
        assert self.enqueue_batch_size >= self.sequence_len * self.delta_time, \
                ('batch size has to be at least equal to sequence length ' + \
                'times delta time')


    # make it each example wise, rather than batch wise
    def apply_filter(self, data):
        sequence_len = tf.constant(self.sequence_len, dtype = tf.int32)
        for f in self.filter.keys:
            data[f] = tf.cast(data[f], tf.bool)
            # Add the batch dimension
            data[f] = tf.expand_dims(data[f], axis=0)
        # combine filters according to specified filter rule
        master_filter = self.filter.eval(data)
        # check if ALL binary labels within sequence are not zero
        master_filter_sum = tf.reduce_sum(tf.cast(master_filter, tf.int32))
        # gather positive examples for each data entry
        return tf.equal(master_filter_sum, sequence_len)


    def enqueue_many_func(self, all_tensors):
        return tf.data.Dataset.zip(
                {key: tf.data.Dataset.from_tensor_slices(value) 
                    for key, value in all_tensors.items()})


    def postprocess_each(self, serialized, source):
        meta = self.meta_dict[source]
        features = {}
        for source_key, meta_data in meta.items():
            if meta_data['dtype'] not in [tf.float32, tf.int64, tf.string]:
                dtype = tf.string
                shape = []
                meta_data['rawtype'] = meta_data['dtype']
                meta_data['rawshape'] = meta_data['shape']
            else:
                shape = meta_data['shape']
                dtype = meta_data['dtype']
            features[source_key] = tf.io.FixedLenFeature(shape, dtype)

        parsed = tf.io.parse_single_example(serialized, features)

        for each_source, curr_data in parsed.items():
            if curr_data.dtype is tf.string:
                curr_meta = self.meta_dict[source][each_source]
                curr_data = tf.io.decode_raw(curr_data, curr_meta['rawtype'])
                curr_data = tf.reshape(curr_data, curr_meta['rawshape'])

            if curr_data.dtype==tf.int16:
                curr_data = tf.cast(curr_data, tf.int32)
            parsed[each_source] = curr_data

        if len(parsed.keys())==1:
            return parsed[list(parsed.keys())[0]]
        else:
            return parsed


    def get_tfrecord_filenames(self, folder_name, file_pattern='*.tfrecords'):
        # Get list of tfrecord filenames for given folder
        tfrecord_pattern = os.path.join(folder_name, file_pattern)
        datasource = tf.io.gfile.glob(tfrecord_pattern)
        datasource.sort()

        return datasource


    def parse_standard_tfmeta(self, path_dict):
        meta_dict = {}
        for source in path_dict:
            path = path_dict[source]
            if isinstance(path, str):
                if path.startswith('meta') and path.endswith('.pkl'):
                    mpaths = [path]
                else:
                    assert os.path.isdir(path), path
                    mpaths = filter(
                            lambda x: x.startswith('meta') \
                                    and x.endswith('.pkl'),
                            os.listdir(path))
                    mpaths = [os.path.join(path, mp) for mp in mpaths]
            else:
                # in this case, it's a list
                assert isinstance(path, list), "Path should be a list"
                mpaths = path
            d = {}
            for mpath in mpaths:
                d.update(pickle.load(open(mpath, 'rb'), encoding='latin1'))
            meta_dict[source] = d
        return meta_dict


    def set_data_shape(self, data):
        shape = data.get_shape().as_list()
        shape[0] = self.enqueue_batch_size
        for s in shape:
            assert s is not None, ("Unknown shape", shape)
        data.set_shape(shape)
        return data


    def pad_up_to(self, tensor, max_shape, constant_values):
        shape = tf.shape(tensor)
        paddings = [[0, tf.maximum(m - shape[i], 0)] if m is not None else [0, 0] \
                    for (i, m) in enumerate(max_shape)]
        return tf.pad(tensor, paddings, 'CONSTANT', constant_values=constant_values)


    def create_data_sequence(self, data):
        if self.use_legacy_sequence_mode and self.delta_time==1:
            data = tf.expand_dims(data, 1)
            data_shape = data.get_shape().as_list()
            data_type = data.dtype
            shift_len = self.enqueue_batch_size - (self.sequence_len - 1)
            shifts = [data[i : i + shift_len] \
                    for i in range(self.sequence_len)]

            shifts = tf.concat(shifts, axis = 1)
            if self.shift_selector:
                # Use only first shift during evaluation
                shifts = shifts[self.shift_selector]
            return shifts
        else:
            data = tf.expand_dims(data, 0)
            sequences = [data[:, i : i+self.sequence_len*self.delta_time : \
                    self.delta_time] for i in \
                    range(self.enqueue_batch_size - (self.sequence_len - 1) * \
                        self.delta_time)]
            return tf.concat(sequences, axis = 0)


    def build_one_dataset(self, curr_data):
        # Unpack the data related info, num_examples is not used
        curr_data_path = curr_data

        # Dictionary with keys being source, and values being directories
        self.source_paths = { 
                source: os.path.join(curr_data_path, source) \
                for source in self.sources }

        # load filters, add that to source_paths
        if self.filter_rule:
            self.filter = Filter(self.filter_rule)
            for f in self.filter.keys:
                self.source_paths[f] = os.path.join(curr_data_path, f)
                if f not in self.all_sources:
                    self.all_sources.append(f)
        else:
            self.filter = None

        # load metas
        self.meta_dict = self.parse_standard_tfmeta(self.source_paths)

        # Get tfr filenames
        source_lists = {
                source: self.get_tfrecord_filenames(
                    self.source_paths[source], 
                    file_pattern=self.file_pattern) \
                for source in self.source_paths}

        # This shuffle needs to be False to keep the order of every attribute
        # the same
        file_datasets = {
                source: tf.data.Dataset.list_files(curr_files, shuffle=False, seed=self.seed) \
                for source, curr_files in source_lists.items()}

        #TODO This disables file shuffling for distributed training. NEED TO FIX SHUFFLING!
        '''
        if self.shuffle and False:
            # Shuffle file names using the same seed
            file_datasets = {
                    source: curr_dataset.shuffle(
                        buffer_size=len(list(source_lists.values())[0]),
                        seed=self.seed).repeat() \
                    for source,curr_dataset in file_datasets.items()}
        '''

        # Create dataset for both
        def _fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024     # 8 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        each_dataset = {
                source: curr_dataset.interleave(
                    _fetch_dataset,
                    cycle_length=1,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                            for source,curr_dataset in file_datasets.items()
                }

        # Decode raw first before zip
        each_dataset = {
                source: curr_dataset.map(
                    lambda x: self.postprocess_each(x, source),
                    num_parallel_calls=self.map_pcall_num,
                    ) \
                for source, curr_dataset in each_dataset.items()
                }

        # Zip, repeat, batch
        zip_dataset = tf.data.Dataset.zip(each_dataset)
        def _expand_group_keys(value):
            new_value = {}
            for key, curr_value in value.items():
                if key in self.subsources:
                    new_value.update(curr_value)
                else:
                    new_value[key] = curr_value
            return new_value

        zip_dataset = zip_dataset.map(
                _expand_group_keys, 
                num_parallel_calls=self.map_pcall_num,
                )

        if self.repeat:
            zip_dataset = zip_dataset.repeat()
        else:
            # Make sure data is used only once during validation
            zip_dataset = zip_dataset.repeat(1)
        zip_dataset = zip_dataset.batch(self.enqueue_batch_size)

        # Set shape (first dimension to be batchsize)
        zip_dataset = zip_dataset.map(
                lambda x: {
                    key: self.set_data_shape(value) 
                    for key,value in x.items()}, 
                num_parallel_calls=self.map_pcall_num)

        # Create sequence for each dataset
        zip_dataset = zip_dataset.map(
                lambda x: {
                    key: self.create_data_sequence(value) 
                    for key, value in x.items()}, 
                num_parallel_calls=self.map_pcall_num)

        return zip_dataset


    def get_max_shapes(self, zip_datasets):
        max_shapes = {}

        for each_dataset in zip_datasets:
            # TODO: Replace with 2.0 api
            curr_shapes = tf.compat.v1.data.get_output_shapes(each_dataset)
            for source, curr_shape in curr_shapes.items():
                curr_shape = curr_shape.as_list()
                if source not in max_shapes:
                    max_shapes[source] = curr_shape
                assert len(max_shapes[source]) == len(curr_shape), \
                        "Length of shapes should be the same! " \
                        + str(source) + " " + str(curr_shape) \
                        + ", " + str(max_shapes[source])

                max_shapes[source] = list(np.maximum( \
                        max_shapes[source], \
                        curr_shape))

        return max_shapes


    def pad_tensors(self, zip_datasets):
        max_shapes = self.get_max_shapes(zip_datasets)

        def _pad_to_max_shapes(value):
            for source, max_shape in max_shapes.items():
                mask_key = source + '_mask'
                assert mask_key not in value, "%s mask already found!" % mask_key
                if source in value:
                    value[mask_key] = self.pad_up_to(
                            tf.ones(tf.shape(value[source]), dtype=tf.bool),
                            max_shape, 0)
                    value[mask_key].set_shape(max_shape)
                    value[source] = self.pad_up_to(value[source], max_shape, 0)
                    value[source].set_shape(max_shape)
                else:
                    if self.key_completion:
                        # TODO: Find better way to deal with missing keys, especially
                        # dtype of source (tf.int32 should not be hard coded here)
                        value[mask_key] = tf.zeros(max_shape, dtype = tf.bool)
                        value[mask_key].set_shape(max_shape)
                        value[source] = tf.zeros(max_shape, dtype = tf.int32)
                        value[source].set_shape(max_shape)
                        print("WARNING: Key %s not found in data!" % source)
                    else:
                        raise KeyError('Key %s not found in data!' % source)

                if mask_key not in self.all_sources:
                    self.all_sources.append(mask_key)
            return value

        for idx in range(len(zip_datasets)):
            zip_datasets[idx] = zip_datasets[idx].map(
                    _pad_to_max_shapes,
                    num_parallel_calls=self.map_pcall_num)
        return zip_datasets


    def concate_datasets(self, zip_datasets):
        zip_dataset = tf.data.Dataset.zip(tuple(zip_datasets))

        def _concate(*value):
            new_value = {}
            all_sources = value[0].keys()
            for source in all_sources:
                new_value[source] = []
                for _each_value in value:
                    new_value[source].append(_each_value[source])
                new_value[source] = tf.concat(new_value[source], axis=0)
            return new_value
        zip_dataset = zip_dataset.map(
                _concate,
                num_parallel_calls=self.map_pcall_num)
        return zip_dataset


    def build_datasets(self, batch_size):
        # Build dataset for every data path
        zip_datasets = [
                self.build_one_dataset(curr_data)\
                for curr_data in self.data]

        # Pad and concatenate
        zip_datasets = self.pad_tensors(zip_datasets)
        zip_dataset = self.concate_datasets(zip_datasets)

        # "Enqueue_many" it, shuffle it
        zip_dataset = zip_dataset.flat_map(self.enqueue_many_func)
        # Apply filters
        if self.filter:
            zip_dataset = zip_dataset.filter(self.apply_filter)
        if self.shuffle:
            # Shuffle it
            zip_dataset = zip_dataset.shuffle(
                    buffer_size=self.buffer_size,
                    seed=self.seed,
                    )
        # Batch it again
        zip_dataset = zip_dataset.batch(batch_size, drop_remainder=True)
        zip_dataset = zip_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return zip_dataset


    # entry point for TFUtils
    def input_fn(self, batch_size, params=None, **kwargs):
        self.model_batch_size = batch_size
        zip_dataset = self.build_datasets()
        zip_iter = tf.compat.v1.data.make_one_shot_iterator(zip_dataset)
        input_dict = zip_iter.get_next()
        input_dict['full_particles'] = input_dict.pop(self.main_source_key)
        return input_dict
