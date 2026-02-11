#!/bin/bash
# generates code snippets for blog post

grep -v ' # profile' pft_jax.py > /tmp/clean.py
awk '/^def train_hessian/{p=1; print; next} /^def train_hvp/{p=0; next} p{print}' /tmp/clean.py > /tmp/hessian.py
awk '/^def train_hvp/{p=1; print; next} /^def [a-zA-Z_]/ && p{p=0; next} p{print}' /tmp/clean.py > /tmp/hvp.py

echo "Hessian JAX"
echo '```python'
cat /tmp/hessian.py
echo '```'
echo ""
echo "HVP JAX"
echo '```diff'
diff -U 9999 /tmp/hessian.py /tmp/hvp.py
echo '```'
echo ""
grep -v ' # profile' pft_torch.py > /tmp/clean.py
awk '/^def train_hessian/{p=1; print; next} /^def train_hvp/{p=0; next} p{print}' /tmp/clean.py > /tmp/hessian.py
awk '/^def train_hvp/{p=1; print; next} /^def [a-zA-Z_]/ && p{p=0; next} p{print}' /tmp/clean.py > /tmp/hvp.py

echo "Hessian Torch"
echo '```python'
cat /tmp/hessian.py
echo '```'
echo ""
echo "HVP Torch"
echo '```diff'
diff -U 9999 /tmp/hessian.py /tmp/hvp.py
echo '```'
echo ""