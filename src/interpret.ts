// This file implements the interpreter for this language.
/*
The lexer will have the following tokens:

ARC, LINE, VAR, EXPR, STARTLOOP, ENDLOOP

There are 5 types of “actions”:

Draw an arc:

Arc center-x center-y width height start-angle stop-angle

Draw a line:

Line start-x start-y stop-x stop-y

Assign expr to variable:

x = 5+2

Start loop:

while x>5

End Loop:

endwhile

The lexer will work as follows:

Split each line by whitespace
If the token[0] is while,endwhile,arc or line add them
Else add VAR:token[0] skip next token
The rest of tokens are expressions
The parser and evaluator works like this:

Create the following data structures

ARC : {
	center-x:arg[1],
	center-y:arg[2],
	width:arg[3],
	height:arg[4],
	start-angle:arg[5],
	stop-angle:arg[6]
}



LINE : {
	start-x:arg[1],
	start-y:arg[2],
	stop-x:arg[3],
	stop-y:arg[4],
}



In case of a while loop create a compound structure which includes all expressions in the loop and the terminal condition

Loop:{

EXPRS:[]
TERMINATION:EXPR
}

Each expression maps to a ts function and the interpreter output is a list:
[{type:ARC,data:...},{type:LINE,data:...}]



Tasks:
Setup react-three/fiber with react 
    -Create draw function for ARC and LINE
    -Call draw function from react
    -Add textbox in react to call interpreter on button click
    -Create render function to render the output list of objects

Write version of interpreter to run single command->Line and arc:
    -Create lexer for ARC and LINE
    -Parse tokens to create ARC and LINE objects
    -return ARC and LINE objects

Enhancements:
    Add variables and expressions
    Add loops
    Add error validation
    Add coloring
*/


