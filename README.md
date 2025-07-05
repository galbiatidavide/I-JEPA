# 🧠 I-JEPA: Joint Embedding Predictive Architecture for Image Reconstruction

Scientific Computing Tools for Advanced Mathematical Modelling  
**Authors**: Giovanni Messa, Francesco Montagnani, Davide Galbiati  
**Instructor**: Prof. Stefano Pagani  
**Institution**: Politecnico di Milano  
**Academic Year**: 2023–2024  

---

## 📌 Project Overview

This project investigates **self-supervised image representation learning** using a custom implementation of **Masked AutoEncoders (MAE)** combined with **Vision Transformers (ViT)**.  

Our architecture is inspired by **Meta AI’s I-JEPA**, and is trained and evaluated on the **MNIST** and **Fashion-MNIST** datasets for grayscale 28×28 images.

---

## 🧠 Model Architecture

Implemented from scratch using PyTorch:

### 🔷 Vision Transformer (ViT)

- Patch embedding via `Rearrange` (einops)  
- Transformer blocks: Multi-head Attention + FeedForward + LayerNorm  
- Positional encoding + CLS token (optional)  
- No external libraries like `timm` used

### 🔷 MAE - Masked AutoEncoder

- Learn to reconstruct randomly masked image patches
- Custom encoder-decoder pipeline:
  - Encoder: ViT
  - Decoder: Lightweight Transformer
- Masking ratio: `75%`
- Output: pixel prediction on masked patches
- Loss: **MSE between masked decoder tokens and encoder embeddings**

```python
recon_loss = 0
for i in range(mask_tokens.shape[0]):
    recon_loss += F.mse_loss(mask_tokens[i], masked_decoder_tokens[i])
recon_loss /= mask_tokens.shape[0]
```

---

## 🧪 Training & Evaluation

Datasets used:
- 🟦 MNIST (digits)
- 🟧 Fashion-MNIST (clothing items)

<table>
  <tr>
    <td align="center">
      <strong>MNIST Sample</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/I-JEPA/main/jepa_images/mnist_classical_example.png" width="250">
    </td>
    <td align="center">
      <strong>Fashion-MNIST Sample</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/I-JEPA/main/jepa_images/mnist_fashion_example.png" width="250">
    </td>
  </tr>
</table>

---

## 🧩 MAE Results

<table>
  <tr>
    <td align="center"><strong>Original Image</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/I-JEPA/main/jepa_images/original_image_1.png" width="250">
    </td>
    <td align="center"><strong>MAE Reconstruction</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/I-JEPA/main/jepa_images/reconstructed_image_1.png" width="250">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><strong>Original</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/I-JEPA/main/jepa_images/original_image_2.png" width="250">
    </td>
    <td align="center"><strong>Reconstructed</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/I-JEPA/main/jepa_images/reconstructed_image_2.png" width="250">
    </td>
  </tr>
</table>

---

## 🧰 Repository Structure

- `classes.py` — core modules: ViT, MAE, Attention, Decoder  
- `mae_digits.ipynb` — training MAE on MNIST  
- `mae_fashion.ipynb` — training MAE on Fashion-MNIST  

---

## ⚙️ Technologies

- PyTorch  
- einops  
- ViT & Transformer blocks from scratch  
- MPS GPU support (`torch.device("mps")`)

---

## 📚 References

- He et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*
- Dosovitskiy et al. (2021). *An Image Is Worth 16×16 Words: Transformers for Image Recognition*
- Meta AI (2023). *I-JEPA: Joint Embedding Predictive Architecture*

---

## 🏫 Academic Context

This project was developed within the course:  
**Scientific Computing Tools for Advanced Mathematical Modelling**  
📍 *Prof. Stefano Pagani – Politecnico di Milano*  
👨‍🎓 MSc Mathematical Engineering – A.Y. 2023–2024
