# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/common.py

import inspect
import traceback
from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Dict, Iterable, List, Optional, Tuple, Union

import hydra
import torch
import wrapt
from huggingface_hub import HfApi, HfFolder, ModelFilter, hf_hub_download
from huggingface_hub.hf_api import ModelInfo
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic import __version__
from atommic.core.connectors.save_restore_connector import SaveRestoreConnector
from atommic.core.neural_types.comparison import NeuralTypeComparisonResult
from atommic.core.neural_types.neural_type import NeuralType
from atommic.utils import logging, model_utils

__all__ = ["Typing", "FileIO", "Model", "PretrainedModelInfo", "Serialization", "is_typecheck_enabled", "typecheck"]

_TYPECHECK_ENABLED = True
_HAS_HYDRA = True


def is_typecheck_enabled():
    """Getter method for typechecking state."""
    return _TYPECHECK_ENABLED


@dataclass
class TypecheckMetadata:
    """Metadata class for input/output neural types.

    Parameters
    ----------
    original_types : Dict[str, NeuralType]
        Preserve the dictionary of type information provided.
    ignore_collections : bool
        For backward compatibility, container support can be disabled explicitly using this flag. When set to True, all
         nesting is ignored and nest-depth checks are skipped.
    mandatory_types : Dict[str, NeuralType]
        Sub-dictionary of `original_types` which contains only those types which are mandatory to include when calling
        the function.
    base_types : Dict[str, NeuralType]
        Dictionary of flattened `str: NeuralType` definitions, disregarding the nest level details into appropriate
        arguments.
    container_depth : Dict[str, int]
        Dictionary mapping `str: int` - such that the valid depth of the nest of this neural type is recorded.
    has_container_types : bool
        Bool flag declaring if any of the neural types declares a container nest in its signature.
    is_singular_container_type : bool
        Bool flag declaring if this is a single Neural Type with a container nest in its signature. Required for
        supporting python list expansion in return statement.
    """

    original_types: Dict[str, NeuralType]
    ignore_collections: bool

    mandatory_types: Dict[str, NeuralType] = field(init=False)
    base_types: Dict[str, NeuralType] = field(init=False)

    container_depth: Dict[str, int] = field(init=False)
    has_container_types: bool = field(init=False)
    is_singular_container_type: bool = field(init=False)

    def __post_init__(self):
        """Post init function to compute metadata."""
        has_container_types = any(isinstance(type_val, (list, tuple)) for type_val in self.original_types.values())

        self.has_container_types = has_container_types

        # If only one NeuralType is declared, and it declares a container nest, set to True
        self.is_singular_container_type = self.has_container_types and len(self.original_types) == 1

        # If container nests are declared, flatten the nest into `base_types`
        # Also compute the nest depth for each of the NeuralTypes
        if self.has_container_types:
            self.base_types = {}
            self.container_depth = {}

            for type_key, type_val in self.original_types.items():
                depth = 0
                while isinstance(type_val, (list, tuple)):
                    if len(type_val) > 1:
                        raise TypeError(
                            f"Neural Type `{type_key}`: {type_val} definition contains more than one element when"
                            "declaring the nested container structure.\n"
                            "Please ensure that you have only 1 NeuralType inside of the entire nested structure "
                            "definition."
                        )

                    type_val = type_val[0]
                    depth += 1

                self.base_types[type_key] = type_val
                self.container_depth[type_key] = depth
        else:
            # Otherwise, simply preserve the original_types and set depth of nest to 0.
            self.base_types = self.original_types
            self.container_depth = {type_key: 0 for type_key in self.base_types.keys()}

        # Compute subset of original_types which are mandatory in the call argspec
        self.mandatory_types = {
            type_key: type_val for type_key, type_val in self.base_types.items() if not type_val.optional
        }


class Typing(ABC):
    """An interface which endows module with neural types."""

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable input neural type checks."""
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable output neural type checks."""
        return None

    def _validate_input_types(self, input_types=None, ignore_collections=False, **kwargs):
        """This function does a few things.

            1) It ensures that len(self.input_types <non-optional>) <= len(kwargs) <= len(self.input_types).

            2) For each (keyword name, keyword value) passed as input to the wrapped function:

                - Check if the keyword name exists in the list of valid self.input_types names.

                - Check if keyword value has the `neural_type` property.

                    - If it does, then perform a comparative check and assert that neural types are compatible (SAME
                        or GREATER).

                - Check if keyword value is a container type (list or tuple). If yes, then perform the elementwise
                    test of neural type above on each element of the nested structure, recursively.

        Parameters
        ----------
        input_types : class
            Either the `input_types` defined at class level, or the local function overridden type definition.
        ignore_collections : bool
            For backward compatibility, container support can be disabled explicitly using this flag. When set to True,
             all nesting is ignored and nest-depth checks are skipped.
        kwargs : Dict[str, Any]
            Dictionary of argument_name:argument_value pairs passed to the wrapped function upon call.
        """
        # TODO: Properly implement this
        if input_types is None:
            return
        # Precompute metadata
        metadata = TypecheckMetadata(original_types=input_types, ignore_collections=ignore_collections)

        total_input_types = len(input_types)
        mandatory_input_types = len(metadata.mandatory_types)

        # Allow number of input arguments to be <= total input neural types.
        if len(kwargs) < mandatory_input_types or len(kwargs) > total_input_types:
            raise TypeError(
                f"Number of input arguments provided ({len(kwargs)}) is not as expected. Function has "
                f"{total_input_types} total inputs with {mandatory_input_types} mandatory inputs."
            )

        for key, value in kwargs.items():
            # Check if keys exists in the defined input types
            if key not in input_types:
                raise TypeError(
                    f"Input argument {key} has no corresponding input_type match. "
                    f"Existing input_types = {input_types.keys()}"
                )

                # Perform neural type check
            if hasattr(value, "neural_type") and metadata.base_types[key].compare(value.neural_type) not in (
                NeuralTypeComparisonResult.SAME,
                NeuralTypeComparisonResult.GREATER,
            ):
                error_msg = [
                    f"{input_types[key].compare(value.neural_type)} :",
                    f"Input type expected : {input_types[key]}",
                    f"Input type found : {value.neural_type}",
                    f"Argument: {key}",
                ]
                for i, dict_tuple in enumerate(metadata.base_types[key].elements_type.type_parameters.items()):
                    error_msg.insert(i + 2, f"  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}")
                error_msg.extend(
                    f"  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}"
                    for i, dict_tuple in enumerate(value.neural_type.elements_type.type_parameters.items())
                )

                raise TypeError("\n".join(error_msg))

                # Perform input n dim check
            if hasattr(value, "shape"):
                value_shape = value.shape
                type_shape = metadata.base_types[key].axes
                if type_shape is not None and len(value_shape) != len(tuple(type_shape)):
                    name = key

                    raise TypeError(
                        f"Input shape mismatch occurred for {name} in module {self.__class__.__name__} : \n"
                        f"Input shape expected = {metadata.base_types[name].axes} | \n"
                        f"Input shape found : {value_shape}"
                    )

            elif isinstance(value, (list, tuple)):
                for val in value:
                    # This initiates a DFS, tracking the depth count as it goes along the nested structure.
                    # Initial depth is 1 as we consider the current loop to be the 1st step inside the nest.
                    self.__check_neural_type(val, metadata, depth=1, name=key)

    def _attach_and_validate_output_types(self, out_objects, ignore_collections=False, output_types=None):
        """This function does a few things.
            1) It ensures that len(out_object) == len(self.output_types).
            2) If the output is a tensor (or list/tuple of list/tuple ... of tensors), it
                attaches a neural_type to it. For objects without the neural_type attribute,
                such as python objects (dictionaries and lists, primitive data types, structs),
                no neural_type is attached.
                Note: tensor.neural_type is only checked during _validate_input_types which is
                called prior to forward().

        Parameters
        ----------
        output_types : class
            Either the `output_types` defined at class level, or the local function overridden type definition.
        ignore_collections : bool
            For backward compatibility, container support can be disabled explicitly using this flag. When set to True,
             all nesting is ignored and nest-depth checks are skipped.
        out_objects : Dict[str, Any]
            The outputs of the wrapped function.
        """
        # TODO: Properly implement this
        if output_types is None:
            return
        # Precompute metadata
        metadata = TypecheckMetadata(original_types=output_types, ignore_collections=ignore_collections)
        out_types_list = list(metadata.base_types.items())
        mandatory_out_types_list = list(metadata.mandatory_types.items())

        # First convert all outputs to list/tuple format to check correct number of outputs
        if isinstance(out_objects, (list, tuple)):
            out_container = out_objects  # can be any rank nested structure
        else:
            out_container = [out_objects]

        # If this neural type has a *single output*, with *support for nested outputs*,
        # then *do not* perform any check on the number of output items against the number
        # of neural types (in this case, 1).
        # This is done as python will *not* wrap a single returned list into a tuple of length 1,
        # instead opting to keep the list intact. Therefore len(out_container) in such a case
        # is the length of all the elements of that list - each of which has the same corresponding
        # neural type (defined as the singular container type).
        if metadata.is_singular_container_type:
            pass

        elif len(out_container) > len(out_types_list) or len(out_container) < len(mandatory_out_types_list):
            raise TypeError(
                f"Number of output arguments provided ({len(out_container)}) is not as expected. It should be larger "
                f"than {len(out_types_list)} and less than {len(mandatory_out_types_list)}.\nThis can be either "
                "because insufficient/extra number of output NeuralTypes were provided,or the provided NeuralTypes "
                f"{output_types} should enable container support (add '[]' to the NeuralType definition)"
            )

            # Attach types recursively, if possible
        if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
            # Here, out_objects is a single object which can potentially be attached with a NeuralType
            try:
                out_objects.neural_type = out_types_list[0][1]
            except AttributeError:
                pass

                # Perform output n dim check
            if hasattr(out_objects, "shape"):
                value_shape = out_objects.shape
                type_shape = out_types_list[0][1].axes
                if type_shape is not None and len(value_shape) != len(type_shape):
                    name = out_types_list[0][0]

                    raise TypeError(
                        f"Output shape mismatch occurred for {name} in module {self.__class__.__name__} : \n"
                        f"Output shape expected = {type_shape} | \n"
                        f"Output shape found : {value_shape}"
                    )

        elif metadata.is_singular_container_type:
            depth = 0 if len(out_objects) == 1 and isinstance(out_objects, tuple) else 1
            for res in out_objects:
                self.__attach_neural_type(res, metadata, depth=depth, name=out_types_list[0][0])
        else:
            # If more than one item is returned in a return statement, python will wrap the output with an outer tuple.
            # Therefore, there must be a 1:1 correspondence of the output_neural type (with or without nested
            # structure) to the actual output (whether it is a single object or a nested structure of objects).
            # Therefore, in such a case, we "start" the DFS at depth 0 - since the recursion is being applied on 1
            # neural type : 1 output struct (single or nested output). Since we are guaranteed that the outer tuple
            # will be built by python, assuming initial depth of 0 is appropriate.
            for ind, res in enumerate(out_objects):
                self.__attach_neural_type(res, metadata, depth=0, name=out_types_list[ind][0])

    def __check_neural_type(self, obj, metadata, depth: int, name: str = None):
        """Checks if the object is of the correct type, and attaches the correct NeuralType.

        Parameters
        ----------
        obj : object
            Any python object that can be assigned to a value.
        metadata : object
            TypecheckMetadata object.
        depth : int
            Current depth of the recursion.
        name : str
            Optional name used of the source object, when an error is raised.
        """
        if isinstance(obj, (tuple, list)):
            for elem in obj:
                self.__check_neural_type(elem, metadata, depth + 1, name=name)
            return  # after processing nest, return to avoid testing nest itself

        type_val = metadata.base_types[name]

        # If nest depth doesnt match neural type structure depth, raise an error
        if not metadata.ignore_collections and depth != metadata.container_depth[name]:
            raise TypeError(
                "While checking input neural types,\n"
                "Nested depth of value did not match container specification:\n"
                f"Current nested depth of NeuralType '{name}' ({type_val}): {depth}\n"
                f"Expected nested depth : {metadata.container_depth[name]}"
            )

        if hasattr(obj, "neural_type") and type_val.compare(obj.neural_type) not in (
            NeuralTypeComparisonResult.SAME,
            NeuralTypeComparisonResult.GREATER,
        ):
            raise TypeError(
                f"{type_val.compare(obj.neural_type)} : \n"
                f"Input type expected = {type_val} | \n"
                f"Input type found : {obj.neural_type}"
            )

        # Perform input n dim check
        if hasattr(obj, "shape"):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Input shape mismatch occurred for {name} in module {self.__class__.__name__} : \n"
                    f"Input shape expected = {type_shape} | \n"
                    f"Input shape found : {value_shape}"
                )

    def __attach_neural_type(self, obj, metadata, depth: int, name: str = None):
        """Attach NeuralType to the object.

        Parameters
        ----------
        obj : object
            Any python object that can be assigned to a value.
        metadata : object
            TypecheckMetadata object.
        depth : int
            Current depth of the recursion.
        name : str
            Optional name used of the source object, when an error is raised.
        """
        if isinstance(obj, (tuple, list)):
            for elem in obj:
                self.__attach_neural_type(elem, metadata, depth=depth + 1, name=name)
            return  # after processing nest, return to avoid argument insertion into nest itself

        type_val = metadata.base_types[name]

        # If nest depth doesnt match neural type structure depth, raise an error
        if not metadata.ignore_collections and depth != metadata.container_depth[name]:
            raise TypeError(
                "While attaching output neural types,\n"
                "Nested depth of value did not match container specification:\n"
                f"Current nested depth of NeuralType '{name}' ({type_val}): {depth}\n"
                f"Expected nested depth : {metadata.container_depth[name]}"
            )

        try:
            obj.neural_type = type_val
        except AttributeError:
            pass

        # Perform output n dim check
        if hasattr(obj, "shape"):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Output shape mismatch occurred for {name} in module {self.__class__.__name__} : \n"
                    f"Output shape expected = {type_shape} | \n"
                    f"Output shape found : {value_shape}"
                )


class Serialization(ABC):
    """Base class for serialization."""

    @classmethod
    def from_config_dict(cls, config: "DictConfig", trainer: Optional[Trainer] = None):  # noqa: MC0001
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if _HAS_HYDRA:
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
                config = OmegaConf.create(config)
                OmegaConf.set_struct(config, True)

            config = model_utils.maybe_update_config_version(config)

        # Hydra 0.x API
        if ("cls" in config or "target" in config) and "params" in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        elif "_target_" in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        else:
            instance = None
            prev_error = ""

            # Attempt class path resolution from config `target` class (if it exists)
            if "target" in config:
                # No guarantee that this is a omegaconf class
                target_cls = config["target"]
                imported_cls = None
                try:
                    # try to import the target class
                    imported_cls = model_utils.import_class_by_path(target_cls)
                    # use subclass instead
                    if issubclass(cls, imported_cls):
                        imported_cls = cls
                    if (
                        accepts_trainer :=  # pylint: disable=unused-variable
                        # check if the target class accepts a trainer argument
                        Serialization._inspect_signature_for_trainer(imported_cls)
                    ):
                        # Create a dummy PL trainer object
                        instance = imported_cls(cfg=config, trainer=trainer)
                    else:
                        instance = imported_cls(cfg=config)

                except Exception as e:
                    tb = traceback.format_exc()
                    prev_error = f"Model instantiation failed.\nTarget class: {target_cls}\nError: {e}\n{tb}"
                    logging.debug(prev_error + "\n falling back to 'cls'.")
            # target class resolution was unsuccessful, fall back to current `cls`
            if instance is None:
                try:
                    if accepts_trainer := Serialization._inspect_signature_for_trainer(cls):  # noqa: F841
                        instance = cls(cfg=config, trainer=trainer)  # type: ignore
                    else:
                        instance = cls(cfg=config)  # type: ignore
                except Exception as e:
                    # report saved errors, if any, and raise the current error
                    if prev_error:
                        logging.error(f"{prev_error}")
                    raise e from e

        if not hasattr(instance, "_cfg"):
            instance._cfg = config  # pylint: disable=protected-access
        return instance

    def to_config_dict(self) -> "DictConfig":
        """Returns object's configuration to config dictionary"""
        if (hasattr(self, "_cfg")) and (
            self._cfg is not None  # type: ignore  # pylint: disable=access-member-before-definition
        ):
            # Resolve the config dict
            if (_HAS_HYDRA) and (
                isinstance(self._cfg, DictConfig)  # type: ignore  # pylint: disable=access-member-before-definition
            ):
                config = OmegaConf.to_container(
                    self._cfg, resolve=True  # type: ignore  # pylint: disable=access-member-before-definition
                )
                config = OmegaConf.create(config)
                OmegaConf.set_struct(config, True)

                config = model_utils.maybe_update_config_version(config)

            self._cfg = config

            return self._cfg
        raise NotImplementedError(
            "to_config_dict() can currently only return object._cfg but current object does not have it."
        )

    @classmethod
    def _inspect_signature_for_trainer(cls, check_cls):
        """Inspects the signature of the class to see if it accepts a trainer argument."""
        if hasattr(check_cls, "__init__"):
            signature = inspect.signature(check_cls.__init__)
            if "trainer" in signature.parameters:
                return True
        return False


class FileIO(ABC):
    """Base class for file IO."""

    def save_to(self, save_path: str):
        """Standardized method to save a tarfile containing the checkpoint, config, and any additional artifacts.
        Implemented via :meth:`atommic.core.connectors.save_restore_connector.SaveRestoreConnector.save_to`.

        Parameters
        ----------
        save_path : str
            Path to save the checkpoint to.
        """
        raise NotImplementedError()

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Optional[Trainer] = None,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        """Restores model instance (weights and configuration) from a .atommic file.

        Parameters
        ----------
        restore_path : str
            Path to .atommic file from which model should be instantiated.
        override_config_path : str, optional
            Path to .yaml file containing the configuration to override the one in the .atommic file.
        map_location : torch.device, optional
            Device to map the instantiated model to. Default  is ``None``, it will select a GPU if available, falling
            back to CPU otherwise.
        strict : bool, optional
            Passed to load_state_dict. Default is ``True``.
        return_config : bool, optional
            If True, returns the underlying config of the restored model as an OmegaConf DictConfig object without
            instantiating the model.
        trainer : Trainer, optional
            If provided, will be used to instantiate the model.
        save_restore_connector : SaveRestoreConnector, optional
            An optional SaveRestoreConnector object that defines the implementation of the restore_from() method.
        """
        raise NotImplementedError()

    @classmethod
    def from_config_file(cls, path2yaml_file: str):
        """Instantiates an instance of atommic Model from YAML config file. Weights will be initialized randomly.

        Parameters
        ----------
        path2yaml_file : str
            Path to yaml file with model configuration.

        Returns
        -------
        atommic Model instance.
        """
        if issubclass(cls, Serialization):
            conf = OmegaConf.load(path2yaml_file)
            return cls.from_config_dict(config=conf)
        raise NotImplementedError()

    def to_config_file(self, path2yaml_file: str):
        """Saves current instance's configuration to YAML config file. Weights will not be saved.

        Parameters
        ----------
        path2yaml_file : str
            Path to yaml file with model configuration.
        """
        if hasattr(self, "_cfg"):
            self._cfg = model_utils.maybe_update_config_version(self._cfg)  # type: ignore
            with open(path2yaml_file, "w", encoding="utf-8") as fout:
                OmegaConf.save(config=self._cfg, f=fout, resolve=True)
        else:
            raise NotImplementedError()


@total_ordering
@dataclass
class PretrainedModelInfo:
    """Class to store information about a pretrained model."""

    pretrained_model_name: str
    description: str
    location: str
    class_: Union["Model", None] = None
    aliases: Union[List[str], None] = None

    def __repr__(self):
        """Return a string representation of the object."""
        base = self.__class__.__name__
        extras = (
            "pretrained_model_name={pretrained_model_name},\n\t"
            "description={description},\n\t"
            "location={location}".format(**self.__dict__)
        )

        if self.class_ is not None:
            extras = "{extras},\n\t" "class_={class_}".format(extras=extras, **self.__dict__)

        return f"{base}(\n\t{extras}\n)"

    def __hash__(self):
        """Return a hash of the object."""
        return hash(self.location)

    def __eq__(self, other):
        """Return True if self is equal to other."""
        # another object is equal to self, if it's hash is equal to hash(self)
        return hash(self) == hash(other) or self.pretrained_model_name == other.pretrained_model_name

    def __lt__(self, other):
        """Return True if self is less than other."""
        return self.pretrained_model_name < other.pretrained_model_name


class Model(Typing, Serialization, FileIO, ABC):  # type: ignore
    """Abstract class offering interface which should be implemented by all atommic models."""

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        """Should list all pre-trained models available.
        Note: There is no check that requires model names and aliases to be unique. In the case of a collision,
        whatever model (or alias) is listed first in the returned list will be instantiated.

        Returns
        -------
        A list of PretrainedModelInfo entries.
        """
        raise NotImplementedError()

    @classmethod
    def search_huggingface_models(
        cls, model_filter: Optional[Union[ModelFilter, List[ModelFilter]]] = None
    ) -> List[ModelInfo]:
        """Should list all pre-trained models available via Hugging Face Hub.

        The following metadata can be passed via the `model_filter` for additional results.

        .. metadata::
            resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
            limit_results: Optional int, limits the number of results returned.

        .. code-block:: python

            # You can replace <DomainSubclass> with any subclass of ModelPT.
            from atommic.core import ModelPT

            # Get default ModelFilter
            filt = <DomainSubclass>.get_hf_model_filter()

            # Make any modifications to the filter as necessary
            filt.language = [...]
            filt.task = ...
            filt.tags = [...]

            # Add any metadata to the filter as needed
            filt.limit_results = 5

            # Obtain model info
            model_infos = <DomainSubclass>.search_huggingface_models(model_filter=filt)

            # Browse through cards and select an appropriate one
            card = model_infos[0]

            # Restore model using `modelId` of the card.
            model = ModelPT.from_pretrained(card.modelId)

        Parameters
        ----------
        model_filter : Optional ModelFilter or List[ModelFilter] (from Hugging Face Hub)
            Filters the returned list of compatible model cards, and selects all results from each filter.
            Users can then use `model_card.modelId` in `from_pretrained()` to restore a atommic Model.
            If no ModelFilter is provided, uses the classes default filter as defined by `get_hf_model_filter()`.

        Returns
        -------
        list
            A list of ModelInfo entries.
        """
        # Resolve model filter if not provided as argument
        if model_filter is None:
            model_filter = cls.get_hf_model_filter()

        # If single model filter, wrap into list
        if not isinstance(model_filter, Iterable):
            model_filter = [model_filter]

        # Inject `atommic` library filter
        for mfilter in model_filter:
            if isinstance(mfilter.library, str) and mfilter.library != "atommic":
                logging.warning(
                    f"Model filter's `library` tag updated be `atommic`. Original value: {mfilter.library}"
                )
                mfilter.library = "atommic"

            elif isinstance(mfilter, Iterable) and "atommic" not in mfilter.library:  # type: ignore
                logging.warning(
                    "Model filter's `library` list updated to include `atommic`. "
                    f"Original value: {mfilter.library}"  # type: ignore
                )
                mfilter.library = list(mfilter)  # type: ignore
                mfilter.library.append("atommic")  # type: ignore

        # Check if api token exists, use if it does
        is_token_available = HfFolder.get_token() is not None

        # Search for all valid models after filtering
        api = HfApi()

        # Setup extra arguments for model filtering
        all_results = []  # type: List[ModelInfo]

        for mfilter in model_filter:
            cardData = None
            limit = None

            if hasattr(mfilter, "resolve_card_info") and mfilter.resolve_card_info is True:
                cardData = True

            if hasattr(mfilter, "limit_results") and mfilter.limit_results is not None:
                limit = mfilter.limit_results

            results = api.list_models(  # pylint: disable=unexpected-keyword-arg
                filter=mfilter,
                use_auth_token=is_token_available,
                sort="lastModified",
                direction=-1,
                cardData=cardData,
                limit=limit,
            )  # type: List[ModelInfo]

            all_results.extend(results)

        return all_results

    @classmethod
    def get_available_model_names(cls) -> List[str]:
        """Returns the list of model names available. To get the complete model description use
        list_available_models().

        Returns
        -------
        A list of model names.
        """
        return (
            [model.pretrained_model_name for model in cls.list_available_models()]  # type: ignore
            if cls.list_available_models() is not None
            else []
        )

    @classmethod
    def get_hf_model_filter(cls) -> ModelFilter:
        """Generates a filter for HuggingFace models.

        Additionally, includes default values of some metadata about results returned by the Hub.

        .. metadata::
            resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
            limit_results: Optional int, limits the number of results returned.

        Returns
        -------
        list
            A Hugging Face Hub ModelFilter object.
        """
        model_filter = ModelFilter(library="atommic")

        # Attach some additional info
        model_filter.resolve_card_info = False
        model_filter.limit_results = None

        return model_filter

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        refresh_cache: bool = False,
        override_config_path: Optional[str] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Optional[Trainer] = None,
        save_restore_connector: SaveRestoreConnector = None,
        return_only_atommic_model_file_in_cache: bool = False,
    ):
        """Instantiates an instance of atommic. Use restore_from() to instantiate from a local .atommic file.

        Parameters
        ----------
        model_name : str
            The name of the model to instantiate.
        refresh_cache : bool, optional
            If set to True, then when fetching from cloud, this will re-fetch the file from cloud even if it is already
            found in a cache locally.
        override_config_path : str, optional
            Path to a yaml config that will override the internal config file.
        map_location : torch.device, optional
            Optional torch.device() to map the instantiated model to a device. Default is ``None``. It will select a
            GPU if available, falling back to CPU otherwise.
        strict : bool, optional
            Passed to torch.load_state_dict. Default is ``True``.
        return_config : bool, optional
            If set to true, will return just the underlying config of the restored model as an OmegaConf/DictConfig
            object without instantiating the model.
        trainer : Trainer, optional
            Optional Trainer objects to use for restoring the model.
        save_restore_connector : SaveRestoreConnector, optional
            Optional SaveRestoreConnector object to use for restoring the model.
        return_only_atommic_model_file_in_cache : bool, optional
            If set to True, will return the path to the atommic model file in the cache, without instantiating the
            model.

        Returns
        -------
            A model instance of a particular model class or its underlying config (if return_config is set).
        """
        if save_restore_connector is None:
            save_restore_connector = SaveRestoreConnector()

        # Resolve if the pretrained model name is on HF Hub source
        (
            class_,
            atommic_model_file_in_cache,
        ) = cls._get_hf_hub_pretrained_model_info(  # type: ignore
            cls, model_name=model_name, refresh_cache=refresh_cache  # type: ignore
        )
        if return_only_atommic_model_file_in_cache:
            return atommic_model_file_in_cache

        instance, state_dict = class_.restore_from(  # type: ignore
            restore_path=atommic_model_file_in_cache,
            override_config_path=override_config_path,
            map_location=map_location,
            strict=strict,
            return_config=return_config,
            trainer=trainer,
            save_restore_connector=save_restore_connector,
        )
        return instance, state_dict

    def _get_hf_hub_pretrained_model_info(cls, model_name: str, refresh_cache: bool = False) -> Tuple[type, str]:
        """Resolve the HuggingFace Hub model pretrained information given a model name.

        The model name must be of general syntax ``{source_repo}/{model_name}``.

        .. note:
            This allows public, externally contributed models to be run freely using atommic.

        Parameters
        ----------
        model_name : str
            Name of the model. Must be the original name or an alias of the model, without any '/'.
        refresh_cache : bool
            Determines whether cache must be refreshed (model is re-downloaded).

        Returns
        -------
        tuple
            A tuple of details describing :
                -   The resolved class of the model. Since the source is external to atommic, always default to using
                    the calling class. Depend on target class resolution by restore_from() for calling the correct
                    class.
                -   The path to the atommic model (.atommic file) in some cached directory (managed by HF Hub).
        """
        # Resolve the model name without origin for filename
        resolved_username = model_name.split("huggingface.co/")[-1].split("/")[0]
        resolved_model_filename = model_name.split(resolved_username)[-1].split("/blob/main/")[0].strip("/")
        resolved_stem = model_name.split("/")[-1]

        # Check if api token exists, use if it does
        is_token_available = HfFolder.get_token() is not None

        # Try to load the model from the Huggingface Hub
        path = hf_hub_download(
            repo_id=f'{resolved_username}/{resolved_model_filename}',
            filename=resolved_stem,
            library_name="atommic",
            library_version=__version__,
            force_download=refresh_cache,
            use_auth_token=is_token_available,
        )

        # Cannot pre-resolve the specific class without double instantiation (first for config, second for model
        # params). Default to current class, and perform basic class path resolution (handled via restore_from() +
        # target class)
        class_ = cls
        return class_, path  # type: ignore


class typecheck:
    """A decorator which performs input-output neural type checks, and attaches neural types to the output of the
    function that it wraps. Requires that the class inherit from `atommic.core.Typing` in order to perform type
    checking, and will raise an error if that is not the case.

    # Usage (Class level type support)
    .. code-block:: python

        @typecheck()
        def fn(self, arg1, arg2, ...):

    # Usage (Function level type support)
    .. code-block:: python

        @typecheck(input_types=..., output_types=...)
        def fn(self, arg1, arg2, ...):

    Points to be noted:
        1) The brackets () in `@typecheck()` are necessary. You will encounter a TypeError: __init__() takes 1 \
        positional argument but X were given without those brackets.
        2) The function can take any number of positional arguments during definition. When you call this function, \
        all arguments must be passed using kwargs only.
    """

    class TypeState(Enum):
        """
        Placeholder to denote the default value of type information provided.
        If the constructor of this decorator is used to override the class level type definition, this enum value
        indicate that types will be overridden.
        """

        UNINITIALIZED = 0

    def __init__(
        self,
        input_types: Union[TypeState, Optional[Dict[str, NeuralType]]] = TypeState.UNINITIALIZED,
        output_types: Union[TypeState, Optional[Dict[str, NeuralType]]] = TypeState.UNINITIALIZED,
        ignore_collections: bool = False,
    ):
        self.input_types = input_types
        self.output_types = output_types

        self.input_override = input_types != self.TypeState.UNINITIALIZED
        self.output_override = output_types != self.TypeState.UNINITIALIZED
        self.ignore_collections = ignore_collections

    @wrapt.decorator(enabled=is_typecheck_enabled)
    def __call__(self, wrapped, instance: Typing, args, kwargs):
        """Wrapper method that can be used on any function of a class that implements :class:`~atommic.core.Typing`.
        By default, it will utilize the `input_types` and `output_types` properties of the class inheriting Typing.
        Local function level overrides can be provided by supplying dictionaries as arguments to the decorator.
        """
        if instance is None:
            raise RuntimeError("Only classes which inherit atommic.core.Typing can use this decorator !")

        if not isinstance(instance, Typing):
            raise RuntimeError("Only classes which inherit atommic.core.Typing can use this decorator !")

        if hasattr(instance, "input_ports") or hasattr(instance, "output_ports"):
            raise RuntimeError(
                "Typing requires override of `input_types()` and `output_types()`, "
                "not `input_ports() and `output_ports()`"
            )

        # Preserve type information
        if self.input_types is typecheck.TypeState.UNINITIALIZED:
            self.input_types = instance.input_types

        if self.output_types is typecheck.TypeState.UNINITIALIZED:
            self.output_types = instance.output_types

        # Resolve global type or local overridden type
        input_types = self.input_types if self.input_override else instance.input_types
        if self.output_override:
            output_types = self.output_types
        else:
            output_types = instance.output_types

        # If types are not defined, skip type checks and just call the wrapped method
        if input_types is None and output_types is None:
            return wrapped(*args, **kwargs)

        # Check that all arguments are kwargs
        if input_types is not None and len(args) > 0:
            raise TypeError("All arguments must be passed by kwargs only for typed methods")

        # Perform rudimentary input checks here
        instance._validate_input_types(input_types=input_types, ignore_collections=self.ignore_collections, **kwargs)

        # Call the method - this can be forward, or any other callable method
        outputs = wrapped(*args, **kwargs)

        instance._attach_and_validate_output_types(
            output_types=output_types, ignore_collections=self.ignore_collections, out_objects=outputs
        )

        return outputs

    @staticmethod
    def set_typecheck_enabled(enabled: bool = True):
        """Set the global typecheck flag."""
        global _TYPECHECK_ENABLED
        _TYPECHECK_ENABLED = enabled

    @staticmethod
    @contextmanager
    def disable_checks():
        """Temporarily disable type checks."""
        typecheck.set_typecheck_enabled(enabled=False)
        try:
            yield
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
