import matplotlib.pyplot as plt

t=combined_data[combined_data['location']==101]

min_p = 20

particles = t['cell'].unique()

# Create a single plot for particles with more than 20 unique time values
plt.figure(figsize=(12, 7))

for particle in particles:
    particle_data = t[t['cell'] == particle]
    if particle_data['time'].nunique() > min_p:
        plt.plot(particle_data['time'], particle_data['mass'], linestyle='-', linewidth=0.5)

plt.show()