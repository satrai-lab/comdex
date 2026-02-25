#!/usr/bin/env python3
import subprocess
import sys
import time
import os

def run_combined_test(test_name, server_args, client_args, timeout=60):
    server_cmd = [sys.executable, "fl_collab.py"] + server_args
    client_cmd = [sys.executable, "fl_collab.py"] + client_args

    print(f"\n[TEST] {test_name}")
    print("Server command:", " ".join(server_cmd))
    print("Client command:", " ".join(client_cmd))
    try:
        # Start the server process.
        server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(5)  # Wait a few seconds for the server to start.
        # Start the client process.
        client_proc = subprocess.Popen(client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for both processes to finish.
        server_stdout, server_stderr = server_proc.communicate(timeout=timeout)
        client_stdout, client_stderr = client_proc.communicate(timeout=timeout)

        print("\n--- Server Output ---")
        print(server_stdout)
        print(server_stderr)
        print("\n--- Client Output ---")
        print(client_stdout)
        print(client_stderr)

        if server_proc.returncode != 0 or client_proc.returncode != 0:
            print(f"[TEST] {test_name} FAILED (non-zero return code)")
            return False
        else:
            print(f"[TEST] {test_name} PASSED")
            return True
    except subprocess.TimeoutExpired:
        print(f"[TEST] {test_name} TIMED OUT after {timeout} seconds")
        server_proc.kill()
        client_proc.kill()
        return False

def main():
    tests = [
        {
            "name": "MNIST Experiment",
            "server_args": [
                "--role", "server",
                "--membership-approach", "client_driven",
                "--server-address", "localhost:1027",
                "--strategy-name", "fedavg",
                "--job-id", "test_mnist",
                "--num-rounds", "1",
                "--dataset", "mnist",
                "--experiment-mode",
                "--required-participants", "1"
            ],
            "client_args": [
                "--role", "client",
                "--membership-approach", "client_driven",
                "--server-address", "localhost:1027",
                "--dataset", "mnist",
                "--client-id", "mnist_client",
                "--num-clients", "1",
                "--experiment-mode"
            ]
        },
        {
            "name": "HAR Experiment",
            "server_args": [
                "--role", "server",
                "--membership-approach", "client_driven",
                "--server-address", "localhost:1027",
                "--strategy-name", "fedavg",
                "--job-id", "test_har",
                "--num-rounds", "1",
                "--dataset", "har",
                "--experiment-mode",
                "--required-participants", "1",
                "--model-type", "har_cnn"
            ],
            "client_args": [
                "--role", "client",
                "--membership-approach", "client_driven",
                "--server-address", "localhost:1027",
                "--dataset", "har",
                "--client-id", "har_client",
                "--num-clients", "1",
                "--experiment-mode",
                "--model-type", "har_cnn"
            ]
        },
        {
            "name": "Cifar Experiment",
            "server_args": [
                "--role", "server",
                "--membership-approach", "client_driven",
                "--server-address", "localhost:1027",
                "--strategy-name", "fedavg",
                "--job-id", "test_cifar",
                "--num-rounds", "1",
                "--dataset", "cifar10",
                "--experiment-mode",
                "--required-participants", "1",
                "--model-type", "tinycnn"
            ],
            "client_args": [
                "--role", "client",
                "--membership-approach", "client_driven",
                "--server-address", "localhost:1027",
                "--dataset", "cifar10",
                "--client-id", "cifar_client",
                "--num-clients", "1",
                "--experiment-mode",
                "--model-type", "tinycnn"
            ]
        },
    ]

    total = len(tests)
    passed = 0
    for test in tests:
        result = run_combined_test(test["name"], test["server_args"], test["client_args"], timeout=60)
        if result:
            passed += 1
        time.sleep(2)
    print(f"\nSanity check completed: {passed}/{total} tests passed.")

if __name__ == "__main__":
    main()
