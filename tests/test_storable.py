import os 

from rule_estimator.storable import Storable


class ChildStorable(Storable):
    def __init__(self, param1, param2):
        self._store_child_params(level=1)
        
    def wave(self):
        return f"Hi {self.param1} and {self.param2}!"

def test_instantiate_child_storable():
    child = ChildStorable("Annie", "Bob")
    assert isinstance(child, Storable)
    

def test_child_method():
    child = ChildStorable("Annie", "Bob")
    assert child.wave() == 'Hi Annie and Bob!'
    

def test_child_attributes():
    child = ChildStorable("Annie", "Bob")
    assert child.param1 == "Annie"
    assert child.param2 == "Bob"


def test_child_stored_params():
    child = ChildStorable("Annie", "Bob")
    assert isinstance(child._stored_params, dict)
    assert 'param1' in child._stored_params
    assert 'param2' in child._stored_params
    

def test_to_yaml():
    child = ChildStorable("Annie", "Bob")
    assert isinstance(child.to_yaml(), str)

def test_to_code():
    child = ChildStorable("Annie", "Bob")
    assert isinstance(child.to_code(), str)
    assert child.to_code().startswith("ChildStorable")
    

def test_store_yaml():
    child = ChildStorable("Annie", "Bob")
    if os.path.exists("test.yaml"):
        os.remove("test.yaml")
    child.to_yaml("test.yaml")
    assert os.path.exists("test.yaml")
    os.remove("test.yaml")


def test_load_yaml():
    child = ChildStorable("Annie", "Bob")
    if os.path.exists("test.yaml"):
        os.remove("test.yaml")
    child.to_yaml("test.yaml")
    assert os.path.exists("test.yaml")
    child2 = ChildStorable.from_yaml("test.yaml")
    assert child2.wave() == 'Hi Annie and Bob!'
    os.remove("test.yaml")
    