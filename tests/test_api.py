from pettingzoo.test import api_test, parallel_api_test

from orbital import env, env3d, parallel_env, parallel_env3d


def test_aec_api():
    api_test(env(max_steps=20), num_cycles=20)


def test_parallel_api():
    parallel_api_test(parallel_env(max_steps=20), num_cycles=20)


def test_aec_api_3d():
    api_test(env3d(max_steps=20), num_cycles=20)


def test_parallel_api_3d():
    parallel_api_test(parallel_env3d(max_steps=20), num_cycles=20)
