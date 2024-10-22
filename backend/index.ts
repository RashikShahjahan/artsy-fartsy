/*Tasks:



Saving Drawings
    -Create button to save drawings
    -Save drawings to Database
    -Load drawings from Database

Routing and Authentication
    -Create login/register
    -Create protected routes
    -Create logout

News Feed which lets you see all drawings

Enhancements:
    Add coloring[Done]
    Add variables and expressions
    Add loops
    Add error validation

*/

import express from 'express';
import cors from 'cors';
import { interpret } from './interpret';

const app = express();
app.use(cors());
app.use(express.json());
app.post('/interpret', (req, res) => {
    console.log(req.body);
    const { code } = req.body;
    const commands = interpret(code);
    res.json(commands);
});

app.listen(3001, () => {
    console.log('Server is running on port 3001');
});