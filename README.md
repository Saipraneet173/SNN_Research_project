# Energy-Efficient Spiking Neural Networks with ISI Modulation

## 📊 Research Project Overview

This repository contains the implementation of energy-efficient Spiking Neural Networks (SNNs) using Inter-Spike Interval (ISI) modulation, developed as part of a final year research project at Loughborough University (2025). The work explores how temporal coding through ISI-modulated synapses can significantly reduce energy consumption in neural networks while maintaining classification performance.

**Supervisor**: Dr. Shirin Dora  
**Institution**: Loughborough University  
**Submission**: May 2025

## 🎯 Project Motivation

With the rise of edge computing and IoT devices, there's an increasing need for neural networks that can operate under strict energy constraints. This project implements a simplified version of ISI-modulated SNNs (IMSNNs) that adaptively adjust synaptic strength based on the timing patterns of neural spikes, inspired by biological neural systems.

## 🔬 Key Results

Our implementation achieved:
- **33.3% reduction** in spike activity compared to standard SNNs
- **77.02% accuracy** on MNIST dataset
- Demonstrates the fundamental trade-off between energy efficiency and accuracy

While these results show lower performance than the original paper (Adams et al., 2024), they successfully validate the core principle of energy savings through temporal coding under hardware constraints.

## 🛠️ Implementation Details

### Architecture
- **Network Structure**: 784-200-10 (input-hidden-output neurons)
- **Neuron Model**: Leaky Integrate-and-Fire (LIF)
- **Dataset**: MNIST handwritten digits
- **Training**: 15 epochs using Adam optimizer
- **Framework**: PyTorch (implemented as Jupyter notebook)

### Key Features
- ISI-based synaptic modulation using Gaussian functions
- Selective gradient propagation to encourage longer inter-spike intervals
- Poisson rate encoding for input spike generation
- Surrogate gradient approximation for backpropagation

## 📁 Repository Structure

```
energy-efficient-snn/
├── FYP.ipynb                # Main Jupyter notebook with both SNN and IMSNN implementations
├── Research_Paper/
│   ├── FYP_Report.pdf       # Full research paper
└── README.md
```

## 🚀 Getting Started

### Prerequisites
This project is designed to run on **Google Colab** with GPU support. No local installation is required.

### Running on Google Colab (Recommended)

1. **Open Google Colab**: Visit [https://colab.research.google.com/](https://colab.research.google.com/)

2. **Upload the Notebook**: 
   - Upload `FYP.ipynb` to your Google Drive, or
   - Upload directly to Colab via File → Upload notebook

3. **Configure Runtime Environment**:
   - Go to `Runtime → Change runtime type`
   - Set **Hardware accelerator** to `GPU`
   - Recommended: Use T4 GPU (usually default)

4. **Run the Experiments**:
   - Click `Runtime → Run all` to execute both models sequentially
   - First cell runs the IMSNN implementation
   - Second cell runs the conventional SNN for comparison

### Local Installation (Optional)

If you prefer to run locally:
```bash
git clone https://github.com/Saipraneet173/SNN_Research_project.git
cd SNN_Research_project
pip install torch torchvision numpy matplotlib jupyter
jupyter notebook FYP.ipynb
```

## 🖥️ Running the Experiments

The notebook contains two main cells:

1. **Cell 1: IMSNN Implementation** - Trains and evaluates the ISI-modulated SNN
2. **Cell 2: Standard SNN** - Trains and evaluates the baseline SNN for comparison

Simply run all cells sequentially. The notebook will:
- Automatically download the MNIST dataset
- Train both models for 15 epochs
- Display training/test accuracy plots
- Show spike activity visualizations
- Print final evaluation metrics comparing energy efficiency

### Expected Output
- **Training curves**: Accuracy and spike activity across epochs
- **Final metrics**: 
  - SNN: Higher accuracy (~81%) but higher spike count
  - IMSNN: Slightly lower accuracy (~77%) but 33% fewer spikes

## ⚠️ Important Limitations

This implementation represents a **simplified version** of the ISI-modulation mechanism described in Adams et al. (2024):

1. **Neuron-level modulation** instead of synapse-level (due to memory constraints)
2. **Reduced network size** (200 hidden neurons vs. optimal 500)
3. **Limited training duration** (15 epochs due to computational overhead)
4. **Computational complexity** increases training time by ~45%

These constraints were necessary due to hardware limitations but provide a foundation for future improvements.

## 📈 Performance Comparison

| Model | Test Accuracy | Spike Reduction | Training Time |
|-------|--------------|-----------------|---------------|
| Standard SNN | 81.40% | - | 870s |
| IMSNN (Ours) | 77.02% | 33.3% | 1265s |
| IMSNN (Original Paper) | 97.45% | 85.7% | - |

## 📝 Important Notes

- **Reproducibility**: Results are reproducible using Google Colab's T4 GPU environment
- **Training Time**: IMSNN training takes ~45% longer due to ISI modulation computations
- **Variance**: Expect ±1-2% variation in results due to random initialization
- **Documentation**: The notebook includes detailed comments explaining each modification

## 🔄 Future Work

Several promising directions for extending this research:

- Implementing true synapse-level ISI modulation with appropriate hardware
- Deploying on neuromorphic platforms (Intel Loihi, IBM TrueNorth)
- Combining with other efficiency techniques (pruning, quantization)
- Extending to more complex datasets and tasks
- Exploring alternative ISI-modulation functions beyond Gaussian

## 📚 Citation

If you use this code for academic purposes, please cite both this implementation and the original work:

**This Implementation:**
```
@misc{darla2025energy,
  author = {Saipraneet Darla},
  title = {Energy Efficient Spiking Neural Networks},
  year = {2025},
  institution = {Loughborough University},
  note = {Final Year Project, Supervised by Dr. Shirin Dora}
}
```

**Original ISI-Modulation Paper:**
```
@article{adams2024synaptic,
  title={Synaptic modulation using interspike intervals increases energy efficiency of spiking neural networks},
  author={Adams, Dylan and Zajaczkowska, Magda and Anjum, Ashiq and Soltoggio, Andrea and Dora, Shirin},
  journal={arXiv preprint arXiv:2408.02961},
  year={2024}
}
```

## 🤝 Contributing

This project was developed as part of academic research. If you'd like to build upon this work:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes with clear descriptions
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

Particularly welcome are contributions that address the current hardware limitations or implement full synapse-level modulation.

## 📄 License

This project is released under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dr. Shirin Dora for supervision and guidance throughout the project
- Dylan Adams et al. for the original ISI-modulation concept
- Loughborough University for computational resources
- The broader SNN research community for foundational work

## 📧 Contact

For questions about this implementation or research collaboration opportunities, please refer to the research paper in the `/docs` folder or raise an issue in this repository.

---

**Note**: This is a university research project implementation. While functional, it's intended primarily for educational and research purposes. For production use cases, consider the limitations discussed above and explore neuromorphic hardware implementations for optimal performance.
