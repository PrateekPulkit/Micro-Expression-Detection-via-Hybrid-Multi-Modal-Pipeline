# Presentation Script: Hybrid-TCN Masterclass 🎤

**Duration**: 10 Minutes Total | **Voice**: Confident, Expert, Approachable | **Flow**: Sequential

---

### [0:00 - 1:00] Slide 1: The Invisible Signal
**Speaker: Prateek (Lead)**
*(Wait for the 3D title to finish its 'opening spin')*

"Good morning, everyone. And welcome. Today, my team and I aren't just here to show you another AI project. We’re here to show you how we essentially cracked the 'Human Poker Face.' 

In the world of Digital Image Processing, there’s one signal that’s almost impossible to catch: the Micro-Expression. It’s a involuntary flicker of true emotion that happens when we try to hide how we feel. We’ve built a system called **Hybrid-TCN** that sees these signals at a level humans simply can’t. 

Let's dive into why this is the toughest problem in facial recognition."

---

### [1:00 - 2:30] Slides 2-3: The Challenge & The Legend
**Speaker: Srinadh**
*(Right Arrow – Slide 2)*

"So, here’s the problem. You just missed it. Again. A micro-expression lasts for as little as **50 milliseconds**. At standard camera speeds, that’s barely enough frames to even notice a twitch. We call this the 'Subtle Motion Problem.' We're talking about muscle movements smaller than **1.5 pixels**. 

*(Right Arrow – Slide 3)*

Now, back in 2019, a researcher named Polikovsky set the gold standard using Optical Flow and RNNs. He was a legend for his time, but his tools had a 'recurrent bottleneck'—they were sequential. They read video like a slow book, one page at a time. In 2026, we need something that can handle a 200-frame burst without breaking a sweat. That’s where our Hybrid design comes in."

---

### [2:30 - 4:30] Slides 4-5: The Secret Sauce & The Brain
**Speaker: Abhishek**
*(Right Arrow – Slide 4)*

"So, how do we outpace the legend? We don't just look at the face; we track it in three different ways at once. We call this our **Triple-Stream Fusion**. 

We have a Global Scout looking at the whole face, a Local ROI stream focusing on the muscles, and—the most unique part—a Blood Hunter tracking vascular signals. 

*(Right Arrow – Slide 5)*

But the real 'Special Sauce' is the **TCN or Temporal Convolutional Network**. Instead of reading frame-by-frame, our TCN looks at the whole clip in one parallel pass. By using **Dilated Convolutions**, the AI has a 'Zoom Lens' on time. It can see the 'Onset' and the 'Apex' of a twitch simultaneously. It’s faster, it’s snappy, and it doesn't forget."

---

### [4:30 - 6:30] Slides 6-7: Logic, Anatomy & Blood
**Speaker: Tejesh**
*(Right Arrow – Slide 6)*

"But a brain is useless if it doesn't know where to look. In DIP, context is everything. We didn't just throw raw pixels at the AI. We guided its attention using **Action Units**—the specific muscle groups like the *Corrugator Supercilii* that twitch when you're stressed. 

*(Right Arrow – Slide 7)*

And here’s the novelty: **Blood doesn't lie.** You can try to hide a smile, but you can't hide your pulse. We added a stream that tracks the **Cb and Cr chrominance channels** in your skin. When you blink, the motion-sensor might get tricked, but the color-sensor sees no change in blood flow, so it knows to ignore the false alarm. This is the first time Motion and Vascular signals have been fused this effectively."

---

### [6:30 - 8:30] Slides 8-9: Tracking Ghosts & The Proof
**Speaker: Hanoc**
*(Right Arrow – Slide 8)*

"Now, the math behind this is basically 'tracking ghosts.' We use **Farneback Optical Flow** to map out neighborhood shifts between every frame. And to handle the fact that micro-expressions are so rare, we used **Focal Loss**. It basically tells the AI: 'You’ve seen enough normal faces; stop being lazy and find that one-pixel twitch!'

*(Right Arrow – Slide 9)*

Does it work? We didn't just win; we dominated. On the toughest global benchmarks like CASME II, we hit a stable **0.99 F1 Score**. That’s near-perfect precision. We've essentially turned micro-expression detection into a solved problem."

---

### [8:30 - 10:00] Slides 10-11: Deployment & The Mic Drop
**Speaker: Prateek (Lead)**
*(Right Arrow – Slide 10)*

"And the best part? This isn't some supercomputer experiment. It runs right here on this laptop at **24 FPS**. It’s private, it’s on-device, and it uses 96% fewer weights than the old 2019 models. 

*(Right Arrow – Slide 11)*

In conclusion, Hybrid-TCN is the new SOTA. We’ve bridged the gap between medical vascular analysis and high-speed computer vision. We’ve built a system that sees what humans can't. 

Project Complete. We’re ready for your questions. Thank you."

---
**[PRO-TIP]**: Prateek, keep your hand near the Right Arrow key. Hit it exactly when the speakers mention the next slide's concept. The 3D flip takes 0.4s—it’ll look like magic!
