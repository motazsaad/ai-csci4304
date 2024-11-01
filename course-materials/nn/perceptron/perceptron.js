/*
The MIT License (MIT)

Copyright (c) 2014 Richard Teammco

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


/* File: perceptron.js
   Contains code to run the perceptron simulation.
 */


// Global variables:
var RADIUS = 3;
var LEARNING_RATE = 0.5;
var selected = 1;
var points = Array();
var percept = null;
var canvas = null;
var ctx = null;


/* Perceptron object:
   Keeps track of its weights and provides output when given input.
 */
function Perceptron() {
    // weights and bias
    this.bias = 0;
    this.wX = 1;
    this.wY = 1;

    // compute an output value for the given input (tuple of x and y values)
    this.run = function(INPUT) {
        sum = this.bias + (this.wX * INPUT.x) + (this.wY * INPUT.y);
        output = 1 / (1 + Math.exp(-sum)); // sigmoid
        return output;
    }

    // train the perceptron based on its output of training item INPUT
    this.train = function(INPUT, output) {
        var expected = INPUT.cls;
        this.bias = this.bias + LEARNING_RATE * (expected - output);
        this.wX = this.wX + LEARNING_RATE * (expected - output) * INPUT.x;
        this.wY = this.wY + LEARNING_RATE * (expected - output) * INPUT.y;
    }

    // computes the hyperplane (line) y value given the x coordinate.
    // values are assumed to be in scaled (standard) coordinates (-1, +1)
    this.hyperplane = function(x) {
        //( ∑wixi ) + b = 0.
        // W.x (X) + W.y (Y) + b = 0
        // W.y (Y) = -W.x (X) - b
        // Y = (-W.x(X) - b) / W.y
        return ((-this.wX * x) - this.bias) / this.wY;
    }

    // reset the percepron to default values
    this.reset = function() {
        this.bias = 0;
        this.wX = 1;
        this.wY = 1;
    }
}


// get the canvas and draw it for the first time
$(document).ready(function() {
    // set default values
    document.getElementById("learning_rate_box").value = LEARNING_RATE;
    // set up the canvas and perceptron
    canvas = document.getElementById("perceptron_canvas");
    ctx = canvas.getContext("2d");
    percept = new Perceptron();
    drawAll();
});


// set the current class to positive (dots of that class will be added)
function setClass1() {
    selected = 1;
    $("#class1_button").addClass("selected_button");
    $("#class2_button").removeClass("selected_button");
}


// set the current class to negative (dots of that class will be added)
function setClass2() {
    selected = -1;
    $("#class2_button").addClass("selected_button");
    $("#class1_button").removeClass("selected_button");
}


// clear add data from the canvas
function reset() {
    setClass1();
    points = Array();
    percept.reset();
    drawAll();
}


// adjusts the input the be in a standard (-1, +1) coordinate plane
// (translate coordinates from the canvas)... also the y value is 0,
// not -1.
function scaleInput(X) {
    var inputX = (2 * X.x - canvas.width) / canvas.width;
    var inputY = -1 * ((2 * X.y - canvas.height) / canvas.height);
    var inputCls = X.cls == 1 ? 1 : 0;
    INPUT = {x: inputX, y: inputY, cls: inputCls};
    return INPUT;
}


// when the canvas is clicked, add a dot in that position and redraw
function clickHappened(event) {
    // compute point location and class and add it to the system
    var xPos = Math.floor(event.pageX - $(canvas).position().left);
    var yPos = Math.floor(event.pageY - $(canvas).position().top);
    X = {x: xPos, y: yPos, cls: selected, predicted: 0};
    points.push(X);

    // compute the perceptron's expected output and train it
    INPUT = scaleInput(X);
    var output = percept.run(INPUT);
    var color = (output >= 0.5) ? "blue" : "red";
    $("#output_disp").css("color", color);
    $("#output_disp").text(Math.round(output*1000)/1000);
    percept.train(INPUT, output);

    // draw everything
    drawAll();
}


// use the perceptron to compute the hyperplane and translate it back
// to global (canvas) coordinates
function getLineY(x) {
    x = (2*x - canvas.width) / canvas.width;
    var y = percept.hyperplane(x);
    y *= -1;
    y *= canvas.height;
    y += canvas.height;
    y /= 2;
    return y;
}


// draw the canvas (i.e. draw all the dots and the graph)
function drawAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // draw the points
    for(var i=0; i<points.length; i++) {
        var X = points[i];

        ctx.fillStyle = ((X.cls == 1) ? "blue" : "red");
        ctx.beginPath();
        ctx.arc(X.x, X.y, RADIUS, 0, 2*Math.PI);
        ctx.fill();

        // if classified, draw the predicted class this one time
        if(X.predicted != 0) {
            ctx.strokeStyle = ((X.predicted == 1) ? "blue" : "red");
            ctx.beginPath();
            ctx.arc(X.x, X.y, RADIUS*2, 0, 2*Math.PI);
            ctx.stroke();
            X.predicted = 0; // reset prediction
        }
    }

    // draw the graph quadrants
    ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.moveTo(0, canvas.height/2);
    ctx.lineTo(canvas.width, canvas.height/2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(canvas.width/2, 0);
    ctx.lineTo(canvas.width/2, canvas.height);
    ctx.stroke();

    // draw the classification line if possible
    ctx.strokeStyle = "orange";
    ctx.beginPath();
    ctx.moveTo(0, getLineY(0));
    ctx.lineTo(canvas.width, getLineY(canvas.width));
    ctx.stroke();
}


// train the perceptron on all existing points again
// this may increase the accuracy of the classification
function train() {
    for(var i=0; i<points.length; i++) {
        var X = points[i];
        INPUT = scaleInput(X);
        var output = percept.run(INPUT);
        percept.train(INPUT, output);
    }

    drawAll();
}


// same as train, but resets the perceptron first
function retrain() {
    percept.reset();
    train();
}


// run through all the points and have the perceptron classify them visually
function classify() {
    for(var i=0; i<points.length; i++) {
        var X = points[i];
        INPUT = scaleInput(X);
        var output = percept.run(INPUT);
        X.predicted = (output >= 0.5) ? 1 : -1;
    }

    drawAll();
}


// set the learning rate of the perceptron
function setLearningRate() {
    var rate = document.getElementById("learning_rate_box").value;
    if(!isNaN(rate)) {
        rate = parseFloat(rate);
        if(rate >= 0 && rate <= 1)
            LEARNING_RATE = rate;
    }
}
