# Requirements shell script
echo '---requirements shell script---'
#bleeding edge acme
cd /workspace
git clone https://github.com/deepmind/acme.git
cd /workspace/acme
pip install .[jax,tf,testing,envs]
cd /workspace/includes
pip install -r requirements.txt
