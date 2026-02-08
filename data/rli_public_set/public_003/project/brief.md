## Work description

### Overview

Create a casual, web-based game called "Mega Merge" where players combine falling objects to reach the highest-level item possible. The game should be inspired by the popular Watermelon Game but incorporate unique mechanics and features. It should be designed for accessibility and smooth play on any device, with a responsive layout suitable for both desktop and mobile play.

### Objective

Players will aim to combine objects and score as many points as possible before the box fills up. By merging two identical items, players will create a higher-level item. The goal is to manage space strategically while maximizing the score.

The game should have a brewing theme, with the following objects (in order of merging):
1. Water droplet
2. Barley grain
3. Malt bag
4. Orange
5. Lime
6. Hops
7. Pistachio
8. Honey
9. Beer can with open pull tab
10. Mug of beer
11. Barrel of beer

### Key Features

- Platform: Web-based, compatible with all major browsers (Chrome, Safari, Firefox, Edge).
- Cross-Platform Compatibility: Works seamlessly on desktop and mobile (iOS and Android) with responsive layouts.
- Controls: Supports both touch gestures (tap, swipe) and mouse clicks for flexible gameplay.
- Quick Load Times: Lightweight assets and optimized code to ensure rapid load speeds, even on low-end devices.
- Instant Playability: No downloads required; players can start immediately by opening the game in their browser.

### Technical Requirements

- Implementation: The game should be implemented directly using HTML, CSS, and JavaScript. The developer may use a JavaScript framework such as Phaser.js or Construct 3 if it facilitates development and provides necessary physics capabilities. Vanilla JavaScript is also acceptable if it meets all requirements.
- Physics and Collisions: Objects should obey basic physics. They should fall naturally within a defined "box" and exhibit slight bounce effects when landing or colliding with each other. This behavior can be achieved with a physics engine like matter.js or through Construct 3's physics behavior. The collisions boxes for objects should match the object shapes.
- File Size: The total file size should be kept under 5 MB to ensure fast loading.

### Gameplay Mechanics

- Object Merging: Players combine matching items to generate higher-level objects, aiming to reach the final item. Two items combine to form a higher-level item.
- Score Maximization: Each successful merge scores points. Merging high-level items gives more points.
- Limited Space: Players must manage space carefully within a defined container—if the box fills up, the game ends.
- In general, higher-level items should be physically larger than smaller items, but not always.
- If objects cross the top of the box, the round should end. A screen should appear on top of the game saying "Brewing Results" with an indication of which items were reached this round. There should be a button to start a new round.
- Objects should have non-uniform shapes. The collision boxes should match the object shapes.

### Visual & UI Design

- Container Box: Objects should fall into a clearly defined "box" area with visible boundaries, guiding the player's actions.
- Falling Indicator: The next item should have an indicator at the bottom of the screen to show where it will fall, helping players plan their moves.
- Score Display: The score should be displayed prominently at the top of the screen.
- Minimalist UI: Essential elements only—score display, "Next Item" preview, and basic pause/reset buttons at the top of the screen.
- Cartoon-Style Graphics: Use a cute/kawaii, colorful, and cartoonish visual style with simple lines and vibrant colors, similar to the Watermelon Game, for an approachable and relaxing aesthetic.
- Responsive Layout: The design should adapt seamlessly for both desktop and mobile screens.

### Audio and Sound Design

- Background Music: The game should include relaxing background music that plays continuously during gameplay to create a calm, enjoyable atmosphere.
- Sound Effects:
  - A satisfying sound effect should play when objects are dropped into the container.
  - A distinct, gratifying merging sound should play when two objects combine.
  - These sounds should enhance the feedback of each action, creating an engaging and satisfying player experience.

### Interaction and Controls

- Touch and Mouse Support: The game should support both touch gestures and mouse input to provide a smooth experience on both mobile and desktop platforms.
- Responsive Design: The layout should adjust for screen size changes to ensure accessibility and comfort on any device.

## Provided material

None

## Deliverables

Game Files: All files (HTML, CSS, JavaScript, images, and audio files) should be organized in a clear folder structure, with folders for assets, icons, images, scripts, styles, and sounds.