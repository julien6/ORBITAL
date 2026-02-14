from pettingzoo.test import api_test, parallel_api_test

from orbital import env, parallel_env


def test_aec_api():
    api_test(env(max_steps=20), num_cycles=20)


def test_parallel_api():
    parallel_api_test(parallel_env(max_steps=20), num_cycles=20)
