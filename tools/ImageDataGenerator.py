class ImageDataGenerator:
    """DEPRECATED."""

    def __init__(
        self,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()

        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        self.interpolation_order = interpolation_order

        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                "(channel after row and column) or "
                '`"channels_first"` (channel before row and column). '
                f"Received: {data_format}"
            )
        self.data_format = data_format
        if data_format == "channels_first":
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == "channels_last":
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                "`validation_split` must be strictly between 0 and 1. "
                f" Received: {validation_split}"
            )
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.zca_whitening_matrix = None

        if isinstance(zoom_range, (float, int)):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2 and all(
            isinstance(val, (float, int)) for val in zoom_range
        ):
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError(
                "`zoom_range` should be a float or "
                "a tuple or list of two floats. "
                f"Received: {zoom_range}"
            )
        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`zca_whitening`, which overrides "
                    "setting of `featurewise_center`."
                )
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`zca_whitening` "
                    "which overrides setting of"
                    "`featurewise_std_normalization`."
                )
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`featurewise_std_normalization`, "
                    "which overrides setting of "
                    "`featurewise_center`."
                )
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`samplewise_std_normalization`, "
                    "which overrides setting of "
                    "`samplewise_center`."
                )
        if brightness_range is not None:
            if (
                not isinstance(brightness_range, (tuple, list))
                or len(brightness_range) != 2
            ):
                raise ValueError(
                    "`brightness_range should be tuple or list of two floats. "
                    f"Received: {brightness_range}"
                )
        self.brightness_range = brightness_range

    def flow(
        self,
        x,
        y=None,
        batch_size=32,
        shuffle=True,
        sample_weight=None,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        ignore_class_split=False,
        subset=None,
    ):
        return NumpyArrayIterator(
            x,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            ignore_class_split=ignore_class_split,
            subset=subset,
            dtype=self.dtype,
        )

    def flow_from_directory(
        self,
        directory,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
    ):
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            keep_aspect_ratio=keep_aspect_ratio,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            dtype=self.dtype,
        )

    def flow_from_dataframe(
        self,
        dataframe,
        directory=None,
        x_col="filename",
        y_col="class",
        weight_col=None,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        interpolation="nearest",
        validate_filenames=True,
        **kwargs,
    ):
        if "has_ext" in kwargs:
            warnings.warn(
                "has_ext is deprecated, filenames in the dataframe have "
                "to match the exact filenames in disk.",
                DeprecationWarning,
            )
        if "sort" in kwargs:
            warnings.warn(
                "sort is deprecated, batches will be created in the"
                "same order than the filenames provided if `shuffle`"
                "is set to `False`.",
                DeprecationWarning,
            )
        if class_mode == "other":
            warnings.warn(
                '`class_mode="other"` is deprecated, please use '
                '`class_mode="raw"`.',
                DeprecationWarning,
            )
            class_mode = "raw"
        if "drop_duplicates" in kwargs:
            warnings.warn(
                "drop_duplicates is deprecated, you can drop duplicates "
                "by using the pandas.DataFrame.drop_duplicates method.",
                DeprecationWarning,
            )

        return DataFrameIterator(
            dataframe,
            directory,
            self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
            dtype=self.dtype,
        )

    def standardize(self, x):
        """Applies the normalization configuration in-place to a batch of
        inputs.

        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:

        standardize(np.copy(x))

        Args:
            x: Batch of inputs to be normalized.

        Returns:
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= np.std(x, keepdims=True) + 1e-6

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`featurewise_center`, but it hasn't "
                    "been fit on any training data. Fit it "
                    "first by calling `.fit(numpy_data)`."
                )
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= self.std + 1e-6
            else:
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`featurewise_std_normalization`, "
                    "but it hasn't "
                    "been fit on any training data. Fit it "
                    "first by calling `.fit(numpy_data)`."
                )
        if self.zca_whitening:
            if self.zca_whitening_matrix is not None:
                flat_x = x.reshape(-1, np.prod(x.shape[-3:]))
                white_x = flat_x @ self.zca_whitening_matrix
                x = np.reshape(white_x, x.shape)
            else:
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`zca_whitening`, but it hasn't "
                    "been fit on any training data. Fit it "
                    "first by calling `.fit(numpy_data)`."
                )
        return x

    def get_random_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.

        Args:
            img_shape: Tuple of integers.
                Shape of the image that is transformed.
            seed: Random seed.

        Returns:
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(
                    -self.height_shift_range, self.height_shift_range
                )
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(
                    -self.width_shift_range, self.width_shift_range
                )
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2
            )

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(
                -self.channel_shift_range, self.channel_shift_range
            )

        brightness = None
        if self.brightness_range is not None:
            brightness = np.random.uniform(
                self.brightness_range[0], self.brightness_range[1]
            )

        transform_parameters = {
            "theta": theta,
            "tx": tx,
            "ty": ty,
            "shear": shear,
            "zx": zx,
            "zy": zy,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "channel_shift_intensity": channel_shift_intensity,
            "brightness": brightness,
        }

        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.

        Args:
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intensity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        Returns:
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = apply_affine_transform(
            x,
            transform_parameters.get("theta", 0),
            transform_parameters.get("tx", 0),
            transform_parameters.get("ty", 0),
            transform_parameters.get("shear", 0),
            transform_parameters.get("zx", 1),
            transform_parameters.get("zy", 1),
            row_axis=img_row_axis,
            col_axis=img_col_axis,
            channel_axis=img_channel_axis,
            fill_mode=self.fill_mode,
            cval=self.cval,
            order=self.interpolation_order,
        )

        if transform_parameters.get("channel_shift_intensity") is not None:
            x = apply_channel_shift(
                x,
                transform_parameters["channel_shift_intensity"],
                img_channel_axis,
            )

        if transform_parameters.get("flip_horizontal", False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get("flip_vertical", False):
            x = flip_axis(x, img_row_axis)

        if transform_parameters.get("brightness") is not None:
            x = apply_brightness_shift(
                x, transform_parameters["brightness"], False
            )

        return x

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.

        Args:
            x: 3D tensor, single image.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits the data generator to some sample data.

        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.

        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening`
        are set to `True`.

        When `rescale` is set to a value, rescaling is applied to
        sample data before computing the internal data stats.

        Args:
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 4:
            raise ValueError(
                "Input to `.fit()` should have rank 4. Got array with shape: "
                + str(x.shape)
            )
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                "Expected input to be images (as Numpy array) "
                'following the data format convention "'
                + self.data_format
                + '" (channels on axis '
                + str(self.channel_axis)
                + "), i.e. expected either 1, 3 or 4 channels on axis "
                + str(self.channel_axis)
                + ". However, it was passed an array with shape "
                + str(x.shape)
                + " ("
                + str(x.shape[self.channel_axis])
                + " channels)."
            )

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if self.rescale:
            x *= self.rescale

        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype,
            )
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= self.std + 1e-6

        if self.zca_whitening:
            n = len(x)
            flat_x = np.reshape(x, (n, -1))

            u, s, _ = np.linalg.svd(flat_x.T, full_matrices=False)
            s_inv = np.sqrt(n) / (s + self.zca_epsilon)
            self.zca_whitening_matrix = (u * s_inv).dot(u.T)
