import matplotlib.pyplot as plt

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')

# Define the circle positions and labels
circle_positions = [
    (0.5, 0.8, 'Signal Processing Layer\n(Deterministic Processing,\nHigh Signal Accuracy)', 'blue'),
    (0.85, 0.5, 'AI/ML Layer\n(Deep Learning,\nEfficient Models)', 'purple'),
    (0.5, 0.2, 'Energy Efficiency Layer\n(Minimal Energy,\nLow-Power Hardware)', 'green'),
    (0.15, 0.5, 'Real-Time Operation Layer\n(Ultra-Low Latency,\nPredictive Maintenance)', 'red'),
    (0.2, 0.75, 'Software Layer\n(Re-programmability,\nScalable Architectures)', 'yellow'),
    (0.8, 0.75, 'Cost Efficiency Layer\n(Affordable Pricing,\nMass Adoption)', 'orange'),
    (0.5, 0.5, 'Integration Layer\n(AI/DSP Co-design,\nHigh Bandwidth Utilization)', 'cyan')
]

# Draw the circles and labels
for x, y, label, color in circle_positions:
    circle = plt.Circle((x, y), 0.1, color=color, alpha=0.6, edgecolor='black')
    ax.add_artist(circle)
    ax.text(x, y, label, color='black', fontsize=10, ha='center', va='center', wrap=True)

# Add the center label and the image placeholder
ax.text(0.5, 0.5, "Automated Driving Control Loop", fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Draw connecting arrows
arrows = [
    (0.5, 0.7, 0.5, 0.6),  # Between layers (example positioning)
    (0.6, 0.5, 0.5, 0.5),
    (0.5, 0.4, 0.5, 0.3),
    (0.4, 0.5, 0.5, 0.5)
]
for start_x, start_y, end_x, end_y in arrows:
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(facecolor='black', shrink=0.05))

plt.title("Hardware-Software Co-Design for Autonomous Vehicles", fontsize=14)
plt.show()
