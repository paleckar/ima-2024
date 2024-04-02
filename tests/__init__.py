import ast
import inspect
import random
import re
import textwrap
from typing import Any, Callable, Optional, Union
import unittest

import numpy as np


class IMATestCase(unittest.TestCase):

    def __init__(self, methodName: str = '', **params: Any):
        super().__init__(methodName=methodName)
        self.params = params

    @classmethod
    def eval(
            cls,
            tests: Optional[str] = None,
            verbosity: int = 2,
            **params: dict[str, Any]
    ) -> unittest.TestResult:
        class _TestCaseClass(cls):
            def __init__(self, methodName: str = ''):
                super().__init__(methodName=methodName, **params)
        if tests is not None:
            suite = unittest.TestSuite()
            suite.addTests([_TestCaseClass(methodName=t) for t in tests.split(',')])
        else:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(_TestCaseClass)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        return runner.run(suite)

    @staticmethod
    def _ast_node_name(node: ast.AST) -> str:
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            # TODO: probably shouldn't happen
            raise ValueError("This shouldn't happen and probably is a bug in testing")

    @staticmethod
    def inspect_function_dependencies(func: Callable) -> list[str]:
        return [
            IMATestCase._ast_node_name(node.func)
            for node in ast.walk(ast.parse(IMATestCase.get_source(func)))
            if isinstance(node, ast.Call)
        ]

    @staticmethod
    def get_source(object: Any) -> str:
        source = inspect.getsource(object)
        # Remove comments otherwise dedent may not work when commented-out lines are not indented as rest of the code
        source = '\n'.join(l for l in source.splitlines() if not l.strip().startswith('#'))
        while re.match('\s+', source):
            source = textwrap.dedent(source)
        return source
    
    @staticmethod
    def seed_everything(seed: int = 737765) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def assertNotCalling(self, func: Callable, names: list[str]) -> None:
        source = self.get_source(func)

        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):  # check not directly calling forbidden name
                call_name = self._ast_node_name(node.func)
            elif isinstance(node, ast.Assign):  # check not assigning alias to forbidden name
                if isinstance(node.value, (ast.Attribute, ast.Name)):
                    call_name = self._ast_node_name(node.value)
                else:
                    continue
            else:
                continue
            self.assertNotIn(call_name, names)

    def assertCalling(self, func: Callable, names: list[str]) -> None:
        call_names = self.inspect_function_dependencies(func)
        for name in names:
            self.assertIn(name, call_names, msg=f"Function {func.__qualname__} should call {name}")

    def assertNoLoops(self, func: Callable) -> None:
        if any(
            isinstance(e, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
            for e in ast.walk(ast.parse(self.get_source(func)))
        ):
            self.fail(msg=f"Manual loops are not allowed inside {func.__qualname__}")

    def assertArraysClose(
            self,
            actual: Union[float, np.ndarray],
            expected: Union[float, np.ndarray],
            rtol: float = 1e-3,
            atol: float = 1e-4,
            msg: str = ""
    ) -> None:
        self.assertTupleEqual(actual.shape, expected.shape)  # np.allclose broadcasts
        all_close = np.allclose(actual, expected, rtol=rtol, atol=atol)
        if not all_close:
            err_str = f"Arrays are not close!\n"
            if self.params.get('verbose', True):
                err_str += f"\nActual\n{actual}\n"
                err_str += f"\nExpected\n{expected}\n"
            self.fail(msg=f"{msg}\n{err_str}")
