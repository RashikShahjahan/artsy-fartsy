// exports generateCode function which makes an openAI chat completion request to generate code

import { OpenAI } from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const drawingLanguageSpec: string = `
## 1. LINE Command

The LINE command is used to draw a straight line between two points in a 2D plane.

### Syntax:
LINE x1 y1 x2 y2 color

### Parameters:
- x1, y1 (numeric): The coordinates of the start point of the line.
- x2, y2 (numeric): The coordinates of the end point of the line.
- color (string): The color of the line. Must be a valid color string (e.g., "black", "brown", "red").

### Description:
- Draws a line from the point (x1, y1) to the point (x2, y2) using the specified color.

### Example:
LINE 0 0 2 0 black   # Draws a black line from (0,0) to (2,0)
LINE 1 1 1 3 red     # Draws a red vertical line from (1,1) to (1,3)


## 2. ARC Command

The ARC command is used to draw a curved arc between two points on a circle.

### Syntax:
ARC cx cy x1 y1 x2 y2 startAngle endAngle clockwise rotation color

### Parameters:
- cx, cy (numeric): The coordinates of the center of the arc (circle center).
- x1, y1 (numeric): The start point of the arc on the circle’s edge.
- x2, y2 (numeric): The end point of the arc on the circle’s edge.
- startAngle (numeric): The starting angle of the arc, measured in degrees from the positive x-axis.
- endAngle (numeric): The ending angle of the arc, measured in degrees from the positive x-axis.
- clockwise (boolean): Defines the direction of the arc:
  - true: Clockwise direction.
  - false: Counterclockwise direction.
- rotation (numeric): The rotation of the arc's entire coordinate system (in degrees).
- color (string): The color of the arc. Must be a valid color string (e.g., "black", "blue", "green").

### Description:
- Draws an arc between points (x1, y1) and (x2, y2) that follows a circular path defined by the center (cx, cy) and the angles specified.

### Example:
ARC 1 1 0.8 1 1.2 1 180 0 true 0 blue    # Draws a blue semicircle arc centered at (1,1)
ARC 1 1.5 0.85 1.5 1.15 1.5 180 0 true 0 red   # Red semicircular arc (door top)


## 3. General Notes on Colors
- The color parameter accepts any valid string representation of a color:
  - Common color names like "black", "red", "blue", etc.
  - Hexadecimal color codes like "#FF0000" for red.


## 4. Coordinate System
- The coordinate system is a standard 2D Cartesian plane.
  - The x-axis runs horizontally, increasing to the right.
  - The y-axis runs vertically, increasing upwards.


## 5. Clockwise and Counterclockwise Angles in ARC Command
- Clockwise (true): Angles progress clockwise from the starting point.
- Counterclockwise (false): Angles progress counterclockwise from the starting point.
- Angles are measured in degrees, where 0 degrees is along the positive x-axis, 90 degrees is along the positive y-axis, and so on.


## 6. Example Program:

This example draws a house with a base, roof, and an arched door:

# Drawing the base of the house
LINE 0 0 2 0 black   # Bottom of the house
LINE 2 0 2 2 black   # Right side of the house
LINE 2 2 0 2 black   # Top of the house
LINE 0 2 0 0 black   # Left side of the house

# Drawing the roof
LINE 0 2 1 3 brown   # Left side of the roof
LINE 1 3 2 2 brown   # Right side of the roof
LINE 0 2 2 2 brown   # Bottom of the roof

# Drawing an arched door
ARC 1 1.5 0.85 1.5 1.15 1.5 180 0 true 0 red   # Semicircular door arc
`;


export async function generateCode(userPrompt: string) {
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
        { 
            role: "system", 
            content: "You are a code generator for a drawing app. The specifications of the language and you will be generating code for is: " + drawingLanguageSpec + ". You will take a natural language description of a drawing and generate the code to draw that drawing. You will only output valid code in the language speciefied  and nothing else." ,
        },
        {
            role: "user",
            content: userPrompt
        }
    ],
  });
  return response.choices[0].message.content;
}

