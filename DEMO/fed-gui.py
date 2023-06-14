import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import random

class Node:
    def __init__(self, name, ip, port, community):
        self.name = name
        self.ip = ip
        self.port = port
        self.community = community

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()

        self.colors = ["blue", "red", "green", "yellow", "pink", "orange"]
        self.used_colors = []
        self.communities = {}
        self.nodes = {}  # maps node id to node object
        self.node = None
        self.offset_x = 0
        self.offset_y = 0
        self.selected_nodes = []
        self.bridge_button = tk.Button(master, text="Create Bridge", command=self.create_bridge)
        self.bridge_button.grid(row=5, column=1)


    def create_bridge(self):
        # Check that there are 2 nodes selected
        if len(self.selected_nodes) == 2:
            # Draw a line between the nodes
            coords1 = self.canvas.coords(self.selected_nodes[0])
            coords2 = self.canvas.coords(self.selected_nodes[1])
            self.canvas.create_line(coords1[0], coords1[1], coords2[0], coords2[1])

            # Prompt the user for the topic pattern for the bridge
            topic_pattern = simpledialog.askstring("Input", "Enter the topic pattern for the bridge")

            # Update the mosquitto configuration for the nodes
            for node in self.selected_nodes:
                self.generate_config(self.nodes[node], topic_pattern)


    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=500, height=500, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=3)

        self.name_label = tk.Label(self, text="Area Name:")
        self.name_label.grid(row=1, column=0)

        self.name_entry = tk.Entry(self)
        self.name_entry.grid(row=1, column=1)

        self.ip_label = tk.Label(self, text="IP Address:")
        self.ip_label.grid(row=2, column=0)

        self.ip_entry = tk.Entry(self)
        self.ip_entry.grid(row=2, column=1)

        self.port_label = tk.Label(self, text="Port:")
        self.port_label.grid(row=3, column=0)

        self.port_entry = tk.Entry(self)
        self.port_entry.grid(row=3, column=1)

        self.community_label = tk.Label(self, text="Community:")
        self.community_label.grid(row=4, column=0)

        self.community_entry = tk.Entry(self)
        self.community_entry.grid(row=4, column=1)

        self.create_button = tk.Button(self)
        self.create_button["text"] = "Create Node"
        self.create_button["command"] = self.create_node
        self.create_button.grid(row=5, column=0, columnspan=2)
        

    def create_node(self):
        # Create a new node with the given details
        name = self.name_entry.get()
        ip = self.ip_entry.get()
        port = self.port_entry.get()
        community = self.community_entry.get()

        # Simple validation
        if not name or not ip or not port or not community:
            messagebox.showerror("Error", "All fields must be filled in.")
            return

        # Assign a color to the community if it doesn't already have one
        if community not in self.communities:
            color = random.choice([col for col in self.colors if col not in self.used_colors])
            self.communities[community] = color
            self.used_colors.append(color)

        # Create the node
        node = Node(name, ip, port, community)
        
        # Draw a box for the node on the canvas
        id = self.canvas.create_rectangle(50, 50, 100, 100, fill=self.communities[community])
        self.nodes[id] = node  # add node to the nodes dictionary
        #text_id = self.canvas.create_text(75, 75, text=node.name,tags=name)
        
        # Create a dictionary mapping id of rectangle and text to the Node
        self.canvas.tag_bind(id, '<ButtonPress-1>', self.on_press)
        self.canvas.tag_bind(id, '<B1-Motion>', self.on_drag)

        # Associate the text with the rectangle
        #self.canvas.addtag_withtag(id, text_id)

        # Generate the mosquitto config for the node
        self.generate_config(node)

    def on_press(self, event):
        # Record the mouse position and select the current node
        self.offset_x = event.x
        self.offset_y = event.y
        self.node = self.canvas.find_withtag(tk.CURRENT)[0]

        # Add the node to the list of selected nodes
        if len(self.selected_nodes) < 2:
            self.selected_nodes.append(self.node)
        elif len(self.selected_nodes) == 2:
            # If there are already 2 nodes selected, remove the first one and add the new one
            self.selected_nodes.pop(0)
            self.selected_nodes.append(self.node)

    

    def on_drag(self, event):
        # Calculate how far the mouse has moved
        dx = event.x - self.offset_x
        dy = event.y - self.offset_y

        # Move the node by the distance the mouse has moved
        self.canvas.move(self.node, dx, dy)

        # Record the new mouse position
        self.offset_x = event.x
        self.offset_y = event.y




    def generate_config(self, node, topic_pattern=None):
        # Generate a mosquitto config for the given node
        config = f"""
listener {node.port} {node.ip}
allow_anonymous true
"""

        # If a topic pattern is provided, add a bridge configuration
        if topic_pattern:
            config += f"""
connection bridge_to_{node.name}
address {node.ip}:{node.port}
topic {topic_pattern} out 2 "" ""
"""

        # Save to a mosquitto configuration file
        with open(f"{node.name}_mosquitto.conf", "w") as f:
            f.write(config)


root = tk.Tk()
app = Application(master=root)
app.mainloop()