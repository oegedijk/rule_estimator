import os 

import pytest

from rule_estimator.storable import Storable


class ChildStorable(Storable):
    def __init__(self, param1, param2):
        self._store_child_params(level=1)
        
    def wave(self):
        return f"Hi {self.param1} and {self.param2}!"

@pytest.fixture
def child_storable():
    return ChildStorable("Annie", "Bob")


def test_instantiate_child_storable(child_storable):
    assert isinstance(child_storable, Storable)
    
def test_child_method(child_storable):
    assert child_storable.wave() == 'Hi Annie and Bob!'
    
def test_child_attributes(child_storable):
    assert child_storable.param1 == "Annie"
    assert child_storable.param2 == "Bob"

def test_child_stored_params(child_storable):
    assert isinstance(child_storable._stored_params, dict)
    assert 'param1' in child_storable._stored_params
    assert 'param2' in child_storable._stored_params
    
def test_to_yaml(child_storable):
    assert isinstance(child_storable.to_yaml(), str)

def test_to_code(child_storable):
    assert isinstance(child_storable.to_code(), str)
    assert child_storable.to_code().startswith("\nChildStorable")

def test_store_yaml(child_storable):
    if os.path.exists("test.yaml"):
        os.remove("test.yaml")
    child_storable.to_yaml("test.yaml")
    assert os.path.exists("test.yaml")
    os.remove("test.yaml")

def test_load_yaml(child_storable):
    if os.path.exists("test.yaml"):
        os.remove("test.yaml")
    child_storable.to_yaml("test.yaml")
    assert os.path.exists("test.yaml")
    child2 = ChildStorable.from_yaml("test.yaml")
    assert child2.wave() == 'Hi Annie and Bob!'
    os.remove("test.yaml")
    