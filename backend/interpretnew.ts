

// Type definitions for the program structure


type Token = {
    type: string;
    value: string;
    line: number;
}


function lexer(code: string): Token[]{
    const tokens: Token[] = [];
    const lines = code.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === '') continue;
        const tokenValues = line.split(/\s+/);
        for (const tokenValue of tokenValues) {
            if (tokenValue === '') continue;
            else if(tokenValue === 'while'){
                tokens.push({ type: 'WHILE', value: tokenValue, line: i });
            }
            else if (tokenValue === 'line'){
                tokens.push({ type: 'LINE', value: tokenValue, line: i });
            }
            else if (tokenValue === 'arc'){
                tokens.push({ type: 'ARC', value: tokenValue, line: i });
            }
            else if(tokenValue === 'let'){
                tokens.push({ type: 'LET', value: tokenValue, line: i });
            }
            else if(tokenValue === 'endwhile'){
                tokens.push({ type: 'ENDWHILE', value: tokenValue, line: i });
            }
            else{
                tokens.push({ type: 'VAR', value: tokenValue, line: i });
            }
        }
    }
    return tokens;
}


class Parser{
    tokens: Token[];
    tokenIndex: number;
    lineIndex: number;

    constructor(tokens: Token[]){
        this.tokens = tokens;
        this.tokenIndex = 0;
        this.lineIndex = 0;
    }

    parseProgram(){
        if (this.tokens[this.tokenIndex].type === 'WHILE'){
            return this.parseLoop();
        }
        else {
            return this.parseStatement();
        }
    }   
    parseLoop() {
        // consume WHILE
        this.tokenIndex++;
        // parse condition
        const condition = this.parseCondition();
        this.lineIndex++;
        // parse body
        const body = this.parseBody();
        // consume ENDWHILE
        this.tokenIndex++;
        return { condition, body };
    }
    parseBody() {
        while (this.tokens[this.tokenIndex].type !== 'ENDWHILE'){
            this.parseStatement();
            this.tokenIndex++;
        }
    }
    parseCondition() {
        let condition = '';
        while (this.lineIndex === this.tokens[this.tokenIndex].line){
            condition += this.tokens[this.tokenIndex].value;
            this.tokenIndex++;
        }
        return condition;
    }
    parseStatement() {

    }
}



function interpreter(code: string){
    const tokens = lexer(code);
    console.log(tokens);
    //const parser = new Parser(tokens);
    //const commands = parser.parseProgram();
    //return commands;
}

interpreter(`
    while a < 10
    let a:10
    line 10 10 20 20
endwhile
line 10 10 20 20
`);

