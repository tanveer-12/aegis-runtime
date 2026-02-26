import sys
import json
from typing import Dict, Any


def check_python_version() -> Dict[str, Any]:
    """Verify Python version is 3.10+."""
    try:
        version_info = sys.version_info
        is_valid = version_info.major >= 3 and version_info.minor >= 10
        
        return {
            "check": "python_version",
            "status": "pass" if is_valid else "fail",
            "details": {
                "major": version_info.major,
                "minor": version_info.minor,
                "micro": version_info.micro,
                "required": "3.10+",
                "actual": f"{version_info.major}.{version_info.minor}.{version_info.micro}"
            }
        }
    except Exception as e:
        return {
            "check": "python_version",
            "status": "fail",
            "details": {"error": str(e)}
        }


def check_pytorch() -> Dict[str, Any]:
    """Verify PyTorch is installed and importable."""
    try:
        import torch
        return {
            "check": "pytorch",
            "status": "pass",
            "details": {
                "version": torch.__version__,
                "available": True
            }
        }
    except ImportError as e:
        return {
            "check": "pytorch",
            "status": "fail",
            "details": {"error": f"Import failed: {str(e)}"}
        }
    except Exception as e:
        return {
            "check": "pytorch",
            "status": "fail",
            "details": {"error": str(e)}
        }


def check_cuda() -> Dict[str, Any]:
    """Verify CUDA is available through PyTorch."""
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        return {
            "check": "cuda",
            "status": "pass" if cuda_available else "fail",
            "details": {
                "available": cuda_available,
                "version": cuda_version,
                "device_count": device_count
            }
        }
    except ImportError:
        return {
            "check": "cuda",
            "status": "fail",
            "details": {"error": "PyTorch not installed"}
        }
    except Exception as e:
        return {
            "check": "cuda",
            "status": "fail",
            "details": {"error": str(e)}
        }


def check_gpu_properties() -> Dict[str, Any]:
    """Get GPU device properties."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "check": "gpu_properties",
                "status": "fail",
                "details": {"error": "CUDA not available"}
            }
        
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": props.name,
                "total_memory": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            })
        
        return {
            "check": "gpu_properties",
            "status": "pass",
            "details": {"devices": devices}
        }
    except ImportError:
        return {
            "check": "gpu_properties",
            "status": "fail",
            "details": {"error": "PyTorch not installed"}
        }
    except Exception as e:
        return {
            "check": "gpu_properties",
            "status": "fail",
            "details": {"error": str(e)}
        }


def check_transformers() -> Dict[str, Any]:
    """Verify Transformers library is available."""
    try:
        import transformers
        
        return {
            "check": "transformers",
            "status": "pass",
            "details": {
                "version": transformers.__version__,
                "available": True
            }
        }
    except ImportError as e:
        return {
            "check": "transformers",
            "status": "fail",
            "details": {"error": f"Import failed: {str(e)}"}
        }
    except Exception as e:
        return {
            "check": "transformers",
            "status": "fail",
            "details": {"error": str(e)}
        }


def check_nvidia_ml() -> Dict[str, Any]:
    """Verify nvidia-ml-py (pynvml) is available."""
    try:
        import pynvml
    except ImportError as e:
        return {
            "check": "nvidia_ml",
            "status": "fail",
            "details": {"error": f"Import failed: {str(e)}"}
        }

    try:
        pynvml.nvmlInit()
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
        finally:
            pynvml.nvmlShutdown()

        try:
            import importlib.metadata
            pkg_version = importlib.metadata.version("nvidia-ml-py")
        except Exception:
            pkg_version = getattr(pynvml, "__version__", "unknown")

        return {
            "check": "nvidia_ml",
            "status": "pass",
            "details": {
                "version": pkg_version,
                "driver_version": driver_version,
                "available": True
            }
        }
    except Exception as e:
        return {
            "check": "nvidia_ml",
            "status": "fail",
            "details": {"error": str(e)}
        }


def check_gpu_allocation() -> Dict[str, Any]:
    """Test actual GPU tensor allocation."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "check": "gpu_allocation",
                "status": "fail",
                "details": {"error": "CUDA not available"}
            }
        
        # Create small tensor on GPU
        device = torch.device("cuda:0")
        tensor_a = torch.randn(10, 10, device=device)
        tensor_b = torch.randn(10, 10, device=device)
        
        # Perform simple operation (matmul)
        result = torch.matmul(tensor_a, tensor_b)
        
        # Delete tensors
        del tensor_a
        del tensor_b
        del result
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return {
            "check": "gpu_allocation",
            "status": "pass",
            "details": {
                "operation": "matmul",
                "tensor_shape": [10, 10],
                "success": True
            }
        }
    except ImportError:
        return {
            "check": "gpu_allocation",
            "status": "fail",
            "details": {"error": "PyTorch not installed"}
        }
    except Exception as e:
        return {
            "check": "gpu_allocation",
            "status": "fail",
            "details": {"error": str(e)}
        }


def main():
    """Run all checks and output results."""
    # Initialize results list
    results = []
    
    # Run each check function
    checks = [
        check_python_version,
        check_pytorch,
        check_cuda,
        check_gpu_properties,
        check_transformers,
        check_nvidia_ml,
        check_gpu_allocation,
    ]
    
    for check_func in checks:
        result = check_func()
        results.append(result)
    
    # Check if any failed
    all_passed = all(r["status"] == "pass" for r in results)
    
    # Build final output dict
    output = {
        "status": "pass" if all_passed else "fail",
        "checks": results
    }
    
    # Print JSON to stdout
    print(json.dumps(output, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
