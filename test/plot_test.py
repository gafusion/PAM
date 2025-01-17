import matplotlib.pyplot as plt
import pickle

root = {}
with open('out_pam.pkl', 'rb') as f:
    root['OUTPUTS'] = pickle.load(f)

print(list(root['OUTPUTS']['pellet1'].keys()))
plt.plot(root['OUTPUTS']['pellet1'][0.0]['rho_transport'], root['OUTPUTS']['pellet1'][0.0]['nd'])

plt.show()
