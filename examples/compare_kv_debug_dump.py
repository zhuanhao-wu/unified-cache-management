import argparse
import sys

import torch


def iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield "tensor", value
    elif isinstance(value, (list, tuple)):
        for index, tensor in enumerate(value):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Unsupported dumped sequence item type: {type(tensor)}"
                )
            yield str(index), tensor
    else:
        raise TypeError(f"Unsupported dumped value type: {type(value)}")


def compare_tensor(left, right, atol, rtol):
    same_dtype = left.dtype == right.dtype
    result = {
        "left_shape": tuple(left.shape),
        "right_shape": tuple(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "allclose": False,
    }
    if left.shape != right.shape:
        return result

    left = left.float()
    right = right.float()
    diff = (left - right).abs()
    result.update(
        {
            "max_abs_diff": diff.max().item() if diff.numel() else 0.0,
            "mean_abs_diff": diff.mean().item() if diff.numel() else 0.0,
            "allclose": same_dtype
            and torch.allclose(left, right, atol=atol, rtol=rtol),
        }
    )
    return result


def format_diff(value):
    return "n/a" if value is None else f"{value:.8g}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare UCM store-before and load-after KV cache snapshots."
    )
    parser.add_argument("store_dump", help="store_before .pt file")
    parser.add_argument("load_dump", help="load_after .pt file")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    store = torch.load(args.store_dump, map_location="cpu")
    load = torch.load(args.load_dump, map_location="cpu")

    print(f"store phase: {store.get('phase')}, load phase: {load.get('phase')}")
    print(
        f"store rank: {store.get('tp_rank')}, load rank: {load.get('tp_rank')}, "
        f"store request: {store.get('request_id')}, "
        f"load request: {load.get('request_id')}"
    )
    print(f"store blocks: {store.get('blocks')}")
    print(f"load blocks:  {load.get('blocks')}")

    mismatches = 0
    checked = 0
    if store.get("phase") != "store_before":
        print("METADATA_MISMATCH expected store phase=store_before")
        mismatches += 1
    if load.get("phase") != "load_after":
        print("METADATA_MISMATCH expected load phase=load_after")
        mismatches += 1
    if store.get("tp_rank") != load.get("tp_rank"):
        print("METADATA_MISMATCH field=tp_rank")
        mismatches += 1

    common_layers = sorted(set(store["layers"]) & set(load["layers"]))
    missing_layers = sorted(set(store["layers"]) ^ set(load["layers"]))
    for layer_name in missing_layers:
        print(f"MISSING_LAYER layer={layer_name}")
        mismatches += 1

    for layer_name in common_layers:
        store_layer = store["layers"][layer_name]
        load_layer = load["layers"][layer_name]
        common_blocks = sorted(set(store_layer) & set(load_layer))
        missing_blocks = sorted(set(store_layer) ^ set(load_layer))
        for block_id in missing_blocks:
            print(f"MISSING_BLOCK layer={layer_name} block={block_id}")
            mismatches += 1

        for block_id in common_blocks:
            store_tensors = dict(iter_tensors(store_layer[block_id]))
            load_tensors = dict(iter_tensors(load_layer[block_id]))
            missing_tensors = sorted(set(store_tensors) ^ set(load_tensors))
            for tensor_name in missing_tensors:
                print(
                    f"MISSING_TENSOR layer={layer_name} block={block_id} "
                    f"tensor={tensor_name}"
                )
                mismatches += 1

            for tensor_name in sorted(set(store_tensors) & set(load_tensors)):
                result = compare_tensor(
                    store_tensors[tensor_name],
                    load_tensors[tensor_name],
                    args.atol,
                    args.rtol,
                )
                checked += 1
                if not result["allclose"]:
                    mismatches += 1
                print(
                    f"{layer_name} block={block_id} tensor={tensor_name} "
                    f"left_shape={result['left_shape']} "
                    f"right_shape={result['right_shape']} "
                    f"left_dtype={result['left_dtype']} "
                    f"right_dtype={result['right_dtype']} "
                    f"max_abs_diff={format_diff(result['max_abs_diff'])} "
                    f"mean_abs_diff={format_diff(result['mean_abs_diff'])} "
                    f"allclose={result['allclose']}"
                )

    print(f"checked={checked}, mismatches={mismatches}")
    if mismatches:
        print("KV_CHECK_FAILED")
        sys.exit(1)
    print("KV_CHECK_PASSED")


if __name__ == "__main__":
    main()
