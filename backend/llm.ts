// exports generateCode function which makes an openAI chat completion request to generate code

import { OpenAI } from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const drawingLanguageSpec: string = `
# Formal Specification of the Drawing Language

## 1. LINE Command

The LINE command is used to draw a straight line between two points in a 2D plane.

### Syntax:
LINE x1 y1 x2 y2 color

### Parameters:
- x1, y1 (numeric): The coordinates of the start point of the line.
- x2, y2 (numeric): The coordinates of the end point of the line.
- color (string): The color of the line. Must be a valid color string (e.g., "black", "brown", "red").

### Example:
LINE 0 0 2 0 black   # Draws a black line from (0,0) to (2,0)


## 2. ARC Command

The ARC command is used to draw a curved arc between two points on a circle.

### Syntax:
ARC cx cy x1 y1 x2 y2 startAngle endAngle clockwise rotation color

### Parameters:
- cx, cy (numeric): The coordinates of the center of the arc (circle center).
- x1, y1 (numeric): The start point of the arc on the circle's edge.
- x2, y2 (numeric): The end point of the arc on the circle's edge.
- startAngle (numeric): The starting angle of the arc, measured in degrees from the positive x-axis.
- endAngle (numeric): The ending angle of the arc, measured in degrees from the positive x-axis.
- clockwise (boolean): Defines the direction of the arc:
  - true: Clockwise direction.
  - false: Counterclockwise direction.
- rotation (numeric): The rotation of the arc's entire coordinate system (in degrees).
- color (string): The color of the arc. Must be a valid color string (e.g., "black", "blue", "green").

### Example:
ARC 1 1 0 1 2 1 180 0 true 0 red   # Semicircle


## 3. Drawing Basic Shapes:

### 3.1. Circle

A circle can be drawn using two  `ARC` commands to create the top and bottom halves.

#### Example:
ARC 1 1 0 1 2 1 180 0 true 0 red   # Top half of the circle
ARC 1 1 2 1 0 1 0 180 true 0 red   # Bottom half of the circle

### 3.2. Triangle

A triangle can be drawn using three `LINE` commands connecting three vertices.

#### Example:
LINE 0 0 1 2 black   # First side of the triangle
LINE 1 2 2 0 black   # Second side of the triangle
LINE 2 0 0 0 black   # Third side of the triangle (closing the triangle)

### 3.3. Rectangle

A rectangle can be drawn using four `LINE` commands that form opposite sides of equal length.

#### Example:
LINE 0 0 2.5 0 black   # Bottom side of the rectangle 
LINE 2.5 0 2.5 1.5 black   # Right side of the rectangle
LINE 2.5 1.5 0 1.5 black   # Top side of the rectangle
LINE 0 1.5 0 0 black   # Left side of the rectangle (closing the rectangle)

### 3.4. Square

A square is a specific type of rectangle where all sides are equal in length. It can also be drawn using four `LINE` commands.

#### Example:
LINE 0 0 2 0 black   # Bottom side of the square
LINE 2 0 2 2 black   # Right side of the square
LINE 2 2 0 2 black   # Top side of the square
LINE 0 2 0 0 black   # Left side of the square (closing the square)


# Example Program: Complete Scene Using All Shapes

This program draws a simple scene that includes a house, the sun, and a tree, combining a circle, triangle, rectangle, and square

# Drawing the house base (a rectangle)
LINE 0 0 2.5 0 black   # Bottom side of the house (moved left)
LINE 2.5 0 2.5 2 black   # Right side of the house
LINE 2.5 2 0 2 black   # Top side of the house
LINE 0 2 0 0 black   # Left side of the house

# Drawing the house roof (a triangle)
LINE 0 2 1.25 3 black   # Left side of the roof
LINE 1.25 3 2.5 2 black   # Right side of the roof
LINE 0 2 2.5 2 black     # Bottom side of the roof

# Drawing the sun (a circle using two arcs)
ARC 4 4 3 4 5 4 180 0 true 0 yellow   # Top half of the sun
ARC 4 4 5 4 3 4 0 180 true 0 yellow   # Bottom half of the sun

# Drawing a tree trunk (a square)
LINE 3.5 0 4.5 0 brown   # Bottom side of the tree trunk
LINE 4.5 0 4.5 1 brown   # Right side of the tree trunk
LINE 4.5 1 3.5 1 brown   # Top side of the tree trunk
LINE 3.5 1 3.5 0 brown   # Left side of the tree trunk

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

