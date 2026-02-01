"""
Hardware profile definitions.

Reference data for GPU hardware to help users plan deployments.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class GPUType(str, Enum):
    """Common GPU types."""
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    H100_80GB = "H100-80GB"
    H100_SXM = "H100-SXM"
    A10G = "A10G"
    L4 = "L4"
    L40S = "L40S"
    RTX_3090 = "RTX-3090"
    RTX_4090 = "RTX-4090"


@dataclass
class HardwareProfile:
    """
    Hardware profile for a GPU type.
    
    Provides reference data for capacity planning.
    """
    id: str  # e.g., "A100-80GB"
    name: str
    manufacturer: str = "NVIDIA"
    
    # Memory
    vram_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    
    # Compute
    fp16_tflops: float = 0.0
    fp32_tflops: float = 0.0
    tensor_tflops: float = 0.0  # Tensor Core performance
    
    # Architecture
    architecture: str = ""  # e.g., "Ampere", "Hopper"
    cuda_cores: int = 0
    tensor_cores: int = 0
    
    # Power
    tdp_watts: int = 0
    
    # Interconnect
    nvlink_version: Optional[int] = None
    nvlink_bandwidth_gbps: float = 0.0
    pcie_gen: int = 4
    
    # Cloud pricing (approximate)
    aws_hourly_cost: float = 0.0
    gcp_hourly_cost: float = 0.0
    azure_hourly_cost: float = 0.0
    
    # Notes
    notes: str = ""
    best_for: List[str] = None
    
    def __post_init__(self):
        if self.best_for is None:
            self.best_for = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "_id": self.id,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "memory": {
                "vram_gb": self.vram_gb,
                "bandwidth_gbps": self.memory_bandwidth_gbps,
            },
            "compute": {
                "fp16_tflops": self.fp16_tflops,
                "fp32_tflops": self.fp32_tflops,
                "tensor_tflops": self.tensor_tflops,
            },
            "architecture": {
                "name": self.architecture,
                "cuda_cores": self.cuda_cores,
                "tensor_cores": self.tensor_cores,
            },
            "power": {
                "tdp_watts": self.tdp_watts,
            },
            "interconnect": {
                "nvlink_version": self.nvlink_version,
                "nvlink_bandwidth_gbps": self.nvlink_bandwidth_gbps,
                "pcie_gen": self.pcie_gen,
            },
            "cloud_pricing": {
                "aws": self.aws_hourly_cost,
                "gcp": self.gcp_hourly_cost,
                "azure": self.azure_hourly_cost,
            },
            "notes": self.notes,
            "best_for": self.best_for,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareProfile':
        """Create from dictionary."""
        memory = data.get("memory", {})
        compute = data.get("compute", {})
        arch = data.get("architecture", {})
        power = data.get("power", {})
        interconnect = data.get("interconnect", {})
        cloud = data.get("cloud_pricing", {})
        
        return cls(
            id=data.get("_id", ""),
            name=data.get("name", ""),
            manufacturer=data.get("manufacturer", "NVIDIA"),
            vram_gb=memory.get("vram_gb", 0.0),
            memory_bandwidth_gbps=memory.get("bandwidth_gbps", 0.0),
            fp16_tflops=compute.get("fp16_tflops", 0.0),
            fp32_tflops=compute.get("fp32_tflops", 0.0),
            tensor_tflops=compute.get("tensor_tflops", 0.0),
            architecture=arch.get("name", ""),
            cuda_cores=arch.get("cuda_cores", 0),
            tensor_cores=arch.get("tensor_cores", 0),
            tdp_watts=power.get("tdp_watts", 0),
            nvlink_version=interconnect.get("nvlink_version"),
            nvlink_bandwidth_gbps=interconnect.get("nvlink_bandwidth_gbps", 0.0),
            pcie_gen=interconnect.get("pcie_gen", 4),
            aws_hourly_cost=cloud.get("aws", 0.0),
            gcp_hourly_cost=cloud.get("gcp", 0.0),
            azure_hourly_cost=cloud.get("azure", 0.0),
            notes=data.get("notes", ""),
            best_for=data.get("best_for", []),
        )


# Predefined hardware profiles
HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "A100-80GB": HardwareProfile(
        id="A100-80GB",
        name="NVIDIA A100 80GB SXM",
        vram_gb=80.0,
        memory_bandwidth_gbps=2039,
        fp16_tflops=312.0,
        fp32_tflops=19.5,
        tensor_tflops=624.0,
        architecture="Ampere",
        cuda_cores=6912,
        tensor_cores=432,
        tdp_watts=400,
        nvlink_version=3,
        nvlink_bandwidth_gbps=600,
        aws_hourly_cost=4.10,
        best_for=["large-models", "training", "high-throughput"],
    ),
    "H100-80GB": HardwareProfile(
        id="H100-80GB",
        name="NVIDIA H100 80GB SXM",
        vram_gb=80.0,
        memory_bandwidth_gbps=3350,
        fp16_tflops=989.0,
        fp32_tflops=67.0,
        tensor_tflops=1979.0,
        architecture="Hopper",
        cuda_cores=14592,
        tensor_cores=456,
        tdp_watts=700,
        nvlink_version=4,
        nvlink_bandwidth_gbps=900,
        aws_hourly_cost=8.20,
        best_for=["very-large-models", "training", "maximum-performance"],
    ),
    "L4": HardwareProfile(
        id="L4",
        name="NVIDIA L4",
        vram_gb=24.0,
        memory_bandwidth_gbps=300,
        fp16_tflops=121.0,
        fp32_tflops=30.3,
        tensor_tflops=242.0,
        architecture="Ada Lovelace",
        cuda_cores=7424,
        tensor_cores=232,
        tdp_watts=72,
        aws_hourly_cost=0.81,
        best_for=["small-models", "inference", "cost-effective"],
    ),
}

