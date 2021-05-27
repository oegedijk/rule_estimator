__all__ = ['Storable']

import sys
from importlib import import_module
from pathlib import Path
from typing import Union, List, Dict, Tuple

import oyaml as yaml


def encode_storables(obj):
    """replaces all storable instances (child classes of Storable that
    have a ._stored_params attribute) in obj with a dict specifying
    module, name and params. In conjunction with decode_storables(),
    this allows instances with storable attributes to be stored to and
    loaded from yaml.

    Works recursively through sub-list and sub-dicts.
    """
    if hasattr(obj, "_stored_params"):
        if hasattr(obj, "__rulerepr__"):
            return dict(__businessrule__=dict(
                module=obj.__class__.__module__,
                name=obj.__class__.__name__,
                description=obj.__rulerepr__(),
                params=encode_storables(obj._stored_params)))
        else:
            return dict(__businessrule__=dict(
                module=obj.__class__.__module__,
                name=obj.__class__.__name__,
                params=encode_storables(obj._stored_params)))       
    if isinstance(obj, dict):
        return {k:encode_storables(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_storables(o) for o in obj]
    return obj


def decode_storables(obj):
    """replaces all dict-encoded storables in obj with the appropriate function

    Works recursively through sub-list and sub-dicts"""
    if isinstance(obj, dict) and '__businessrule__' in obj:
        obj = obj['__businessrule__']
        cls = getattr(import_module(obj['module']), obj['name'])
        return cls(**decode_storables(obj['params']))
    if isinstance(obj, dict):
        return {k:decode_storables(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_storables(o) for o in obj]
    return obj


def encode_storables_to_python_code(obj, tabs=0)->str:
    """Outputs python code needed to generate a given Storable object.
    Replaces all storable instances (child classes of Storable that
    have a ._stored_params attribute) in into their their name and params.
    Works recursively through sub-list and sub-dicts.
    """
    linestart = "\n" +"\t"*tabs
    indent = "\n" +"\t"*(tabs+1)
    
    if hasattr(obj, "_stored_params"):
        return f"{linestart}{obj.__class__.__name__}({indent}" + f',{indent}'.join([f'{k}={encode_storables_to_python_code(v, tabs+1)}' for k, v in obj._stored_params.items()]) +f"{linestart})"     
    if isinstance(obj, dict):
        return f"{{{', '.join([f'{k}={encode_storables_to_python_code(v, tabs+1)}' for k, v in obj.items()])}}}"
    elif isinstance(obj, list):
        return f"[{', '.join([encode_storables_to_python_code(o, tabs+1) for o in obj])}]"
    elif isinstance(obj, str):
        return f"'{obj}'"
    return str(obj)


class Storable:
    """Parent class that allows child classes to be stored and recovered from a configuration file.
    
    The helper method _store_child_params() can be called in the __init__ and will store all parameters
    as attributes (saving on boiler plate), and also add them to a ._stored_params dict. 
    
    The method .to_yaml() recursively finds and combines all ._stored_params dicts and saves them to
    a .yaml file. Any Storable elements are saved with their import module and name, so that they 
    can be re-imported with .from_yaml()
    """

    def _store_child_params(self, level:int=1):
        """ Store parameters as attributes and to a
        self._stored_params dict.

        Args:
            level (int): level of the callstack to descend to in
                order to get the parameters. When calling this method
                in the __init__ of a child class, level=1. When called
                from the child of a child class, level=2, etc.
        """
        if not hasattr(self, '_stored_params'):
            self._stored_params = {}
        child_frame = sys._getframe(level)
        child_args = child_frame.f_code.co_varnames[1:child_frame.f_code.co_argcount]
        child_dict = {arg: child_frame.f_locals[arg] for arg in child_args}

        for name, value in child_dict.items():
            setattr(self, name, value)
            self._stored_params[name] = value

    def to_yaml(self, filepath:Union[Path, str]=None, 
                return_dict:bool=False, comment:str=None)->Union[str, None]:
        """Store object to a yaml format.

        Args:
            filepath: file where to store the .yaml file. If None then just return the
                yaml as a str.
            return_dict: instead of return a yaml str, return the raw dict.

        """
        config_dict = encode_storables(self)
        if return_dict:
            return config_dict

        comment_str = ""
        if comment is not None:
            for line in comment.splitlines():
                comment_str += "# " + line + "\n"

        yaml_str = yaml.dump(config_dict)

        if filepath is not None:
            with open(Path(filepath), "w") as f:
                f.write(comment_str + yaml_str)
        else:
            return comment_str + yaml_str
        #yaml.dump(config_dict, open(Path(filepath), "w"))

    @classmethod
    def from_yaml(cls, filepath:Union[Path, str]=None, config:Union[Dict, str]=None):
        """Instantiate object from a yaml format.

        Args:
            filepath: file where .yaml is stored.
            config: instead of filepath you can also pass a dictionary or yaml formatted
                str to config.

        """
        if config is not None:
            if isinstance(config, dict):
                config = config
            elif isinstance(config, str):
                config = yaml.safe_load(config)
            else:
                raise ValueError("config should either be a dict generated with .to_yaml(return_dict=True)"
                                " or a yaml str generated with .to_yaml()!")

        config = yaml.safe_load(open(str(Path(filepath)), "r"))
        return decode_storables(config)
    
    def to_code(self)->str:
        return encode_storables_to_python_code(self)