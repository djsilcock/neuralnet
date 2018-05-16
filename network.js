var topology = [16,13,6];
var inputs=16;
var outputs=topology[topology.length-1];
var trainingdata=[];
trainingdata[0]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
trainingdata[1]=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
　
var learningrate = 0.8;
var momentum = 0.9;
var sharpness = 1;
var stoperror = 0.0001;
　
var c = document.getElementById("myCanvas");
var ctx = c.getContext("2d");
　
var op = document.getElementById("opCanvas");
var ctxOutput = op.getContext("2d");
　
var i;
var layer;
var nx = 0;
var n1 = 0;
var n2 = 0;
var xpos = 0;
var ypos = 0;
　
var numlayers = topology.length;
var maxneuronsperlayer = findMax(topology);
var neurons = [];
var errors = [];
var toterror=0;
var expectedoutputs = [];
var meanoutputerror = 0;
　
var synapses=[];
var changes=[];
var prevchanges=[];
　
var startoflayer = [];
　
var xgap = 600 / (maxneuronsperlayer + 1);
var ygap = (600 / numlayers) - 10;
var topmargin = 30;
　
initialiseNet();
drawNet();
　
// === KEY FUNCTIONS ===
function findMax(arr){
    var m = -Infinity, i = 0, n = arr.length;
    for (; i != n; ++i) {
        if (arr[i] > m) {
            m = arr[i];
        }
    }
    return m;
}
function squash(value){
	return (1/(1+Math.exp(-sharpness*value)));
}
function deriv(value){
	return (0.1+value*(1-value)); //0.1 is Fahlman's modification to speed up training
}
function synapseColour(value){ //greys light to dark
	value=squash(value); //to 0-1?
	var shades=['d','c','a','8','6','4'];
	var i = Math.floor( value * (shades.length-1) );
	return ('#' + shades[i] + shades[i] + shades[i]);
}
function neuronColour(value){ //maps 0-1 to a shade of yellow (non-linear!)
	var shades=['0','6','8','a','c','d','e','f'];
	var i = Math.floor( value * (shades.length-1) );
	return ('#' + shades[i] + shades[i] + '0');
}
function synapseindex(v1,v2){
	return(v1*(maxneuronsperlayer+1)+v2);
}
function fire(){ //inputs set? ... fires the network
	for (var layer=0;layer<numlayers-1;layer++){
		var nextlayer = layer+1;
		for (var n2 = startoflayer[nextlayer]+1; n2 <= startoflayer[nextlayer] + topology[nextlayer]; n2++) {
		 	neurons[n2] = 0;
		 	for (var n1 = startoflayer[layer]; n1 <= startoflayer[layer] + topology[layer]; n1++) {
		 		if (n2!=startoflayer[nextlayer]){ //synapse not going to a bias neuron...
		 			var si = synapseindex(n1,n2);
					neurons[n2] += neurons[n1] * synapses[si];
		 		}
		 	}
		 	neurons[n2] = squash(neurons[n2]);
		}
	}
}
function calculateoutputerrors(){ //error for each individual output layer
	var thiserror=0;
	for (var nx=0; nx < outputs; nx++){
		var n = startoflayer[numlayers-1] +1 +nx;
		errors[n] = expectedoutputs[nx] - neurons[n]; //each output error
		errors[n] = errors[n] * deriv(neurons[n]); //?why do this. I got it from somewhere...
		// toterror += Math.abs(errors[n]);
		thiserror += Math.pow(errors[n],2);
	}
	toterror += 0.5 * thiserror;
}
function backpropagateerrors(){
	for(layer=numlayers-2; layer>=0; layer--){ //move up the network from outputs to inputs
		var nextlayer = layer+1;
		for (n1=startoflayer[layer]; n1 <= startoflayer[layer] + topology[layer]; n1++){ //loop all layer
			errors[n1] = 0;
			for (n2=startoflayer[nextlayer] +1; n2 <= startoflayer[nextlayer] + topology[nextlayer]; n2++){
				i = synapseindex(n1,n2);
				errors[n1] += errors[n2] * synapses[i];
			}
			errors[n1] = errors[n1] * deriv(neurons[n1]);
		}
	}
}
function updatesynapseweights(){
	for (layer=0;layer<numlayers-1;layer++){
		var nextlayer = layer+1;
		for (n1=startoflayer[layer]; n1 <= startoflayer[layer] + topology[layer]; n1++){ //loop all layer
			// changes[n1]=0;
			for (var n2 = startoflayer[nextlayer]; n2 <= startoflayer[nextlayer] + topology[nextlayer]; n2++) {
				if (n2!=startoflayer[nextlayer]){ //not going to a bias neuron...
					i = synapseindex(n1,n2);
					changes[i] = neurons[n1] * errors[n2]; //prev += but synapses became huge...
					changes[i] = (momentum * prevchanges[i]) + (1-momentum) * changes[i];
					prevchanges[i] = changes[i]; //store for momentum calcs for next cycle
					synapses[i] += learningrate * changes[i];
				}
			}
		}
	}
}
　
function debug(txt){
	document.getElementById("debug").innerHTML = txt;
}
　
// === SETUP ===
function initialiseNet(){
	for (layer = 0; layer < numlayers; layer++) {
		for (n1 = 0; n1 <= topology[layer]; n1++) {
			if (n1 == 0){
				neurons[nx] = 1; //bias neurons = 1
				errors[nx] = 0;
				startoflayer[layer]=nx;
			} else {
				neurons[nx] = 0; //others = 0
				errors[nx] = 0;
			}
			nx ++;
		}
	}
	randomiseSynapses();
}
　
function randomiseSynapses(){
	//=== randomise synapses
	var n1 = 0;
	for (layer = 0; layer < numlayers-1; layer++) { //all except output layer
		var nextlayer = layer+1;
		for (var n = 0; n <= topology[layer]; n++) {
			for (var n2 = startoflayer[nextlayer]; n2 <= startoflayer[nextlayer] + topology[nextlayer]; n2++) {
				i = synapseindex(n1,n2);
				var randval=0.2 + 0.5 * Math.random();
				if (Math.random() < 0.5){randval = 0 - randval;}
				synapses[i]=randval;
				changes[i]=0;
				prevchanges[i]=0;
			}
			n1++;
		}
	}
}
　
function shuffleTrainingarray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}
　
function runOneEpoch(){
	shuffleTrainingarray(trainingdata);
	toterror=0;
	for (var i=0;i<trainingdata.length;i++){
		var arr=trainingdata[i];
		for(var index=0; index<inputs; index++){
			neurons[index+1]=parseFloat(arr[index]);
		}
		for (index=0;index<outputs;index++){
			expectedoutputs[index]=parseFloat(arr[inputs+index]);
		}
		fire();
		calculateoutputerrors();
		backpropagateerrors();
		updatesynapseweights();
	}
	// toterror = toterror / topology[numlayers-1];
	return(toterror);
}
　
function showNetTest(){
	var results='';
	for (var i=0;i<trainingdata.length;i++){
		arr=trainingdata[i];
		for (var index=0;index<inputs;index++){
			neurons[index+1]=arr[index];
		}
		for (var index=0;index<outputs;index++){
			expectedoutputs[index]=arr[inputs+index];
		}
		fire();
		for (var n=startoflayer[0] +1; n <= startoflayer[0] + topology[0]; n++){
			results += neurons[n].toFixed(0).toString() + ', ';
		}
		results += ' = ';
		for (var n=startoflayer[numlayers-1] +1; n <= startoflayer[numlayers-1] + topology[numlayers-1]; n++){
			results += neurons[n].toFixed(0).toString() + ', ';
		}
		results += '<br />';
	}
	trainedResults.innerHTML = "<hr />Testing trained net...<br />"+results;
}
　
function numberWithCommas(x) {
  return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
　
function reset(){
	initialiseNet();
	drawNet();
}
　
function trainMe(){
	randomiseSynapses();
	var training = 1; //bool
	toterror=0;
	meanoutputerror = 0;
	var epochx = 0;
	var errorOutput = document.getElementById("meanerror");
	var epochCounter = document.getElementById("epochs");
	ctxOutput.clearRect(0,0,800,100);
	var outputX=0;
	var thisY=0;
	var prevY=0;
　
	if (epochx==0){ //start training loop
		var loop = setInterval(function() { // 1ms outer loop
			for (var innerx=0;innerx<100;innerx++){ //run 10 inner loops in that millisecond
				if (training==1){
					meanoutputerror = runOneEpoch();
					epochx++;
					if (epochCounter != null){epochCounter.innerHTML = numberWithCommas(epochx); }
					if (epochx % 10 == 0){ //update every 10 epochs
						var borderWidth = (1350 * meanoutputerror).toFixed(0);
						if (borderWidth > 90){
							borderWidth=90;
						}
						if (errorOutput != null){ errorOutput.innerHTML = (13 * meanoutputerror).toFixed(2); }
　
						thisY=100-borderWidth;
						ctxOutput.clearRect(outputX+1,0,60,100);
						ctxOutput.strokeStyle = 'Pink';
						ctxOutput.beginPath();
						ctxOutput.moveTo(outputX,100);
						ctxOutput.lineTo(outputX,thisY);
						ctxOutput.stroke();
						ctxOutput.strokeStyle = 'Red';
						ctxOutput.beginPath();
						ctxOutput.moveTo(outputX,thisY);
						ctxOutput.lineTo(outputX-1,prevY);
						ctxOutput.stroke();
　
						outputX++;
						prevY=thisY;
						if (outputX==800){outputX=0;}
　
						topMargin = 100 - borderWidth;
					}
					if (meanoutputerror < stoperror || epochx >= 50000){
						training=0;
						clearInterval(loop);
　
						if (trained != null && meanoutputerror < stoperror){
							trained.innerHTML = 'Trained!';
							trained.style.backgroundColor = "lime";
						} else {
							trained.innerHTML = 'Failed to train';
							trained.style.backgroundColor = "pink";
						}
						if (epochCounter != null){epochCounter.innerHTML = numberWithCommas(epochx);}
						if (errorOutput != null){errorOutput.innerHTML = meanoutputerror.toFixed(2);}
						showNetTest();
					}
				}
			}
			drawNet();
		}, 1);
	}
}
　
function drawNet(){
	ctx.clearRect(0,0,800,600);
　
	// === draw synapses
	n1 = 0;
	for (layer=0;layer<numlayers-1;layer++){
		var nextlayer = layer+1;
		ypos = topmargin + layer * ygap;
		var leftmargin = 10 + (maxneuronsperlayer - topology[layer]) * xgap / 2;
		var leftmarginnextlayer = 10 + (maxneuronsperlayer - topology[nextlayer]) * xgap / 2;
		for (var n = 0; n <= topology[layer]; n++) {
			var xpos = leftmargin + n * xgap;
			for (var n2 = startoflayer[nextlayer]; n2 <= startoflayer[nextlayer] + topology[nextlayer]; n2++) {
				var nextlayerxpos = leftmarginnextlayer + (n2-startoflayer[nextlayer]) * xgap;
				if (n2!=startoflayer[nextlayer]){ //not going to a bias neuron...
					i = synapseindex(n1,n2);
					// ctx.strokeStyle = doShade(synapses[i]);
					ctx.strokeStyle = synapseColour(synapses[i]);
					ctx.beginPath();
					ctx.moveTo(xpos,ypos);
					ctx.lineTo(nextlayerxpos,ypos + ygap);
					ctx.stroke();
					ctx.fillStyle = 'LightGreen';
					ctx.font = "12px Arial";
					ctx.fillText( synapses[i].toFixed(3), xpos - 8 + (nextlayerxpos-xpos)/4, ypos + (3+n2-startoflayer[nextlayer]) * 12);
				}
			}
			n1 ++;
		}
	}
　
	// === draw neurons
	nx = 0;
	ctx.strokeStyle="black";
	for (layer=0;layer<numlayers;layer++){
		ypos = topmargin + layer * ygap;
		var leftmargin = 10 + (maxneuronsperlayer - topology[layer]) * xgap / 2;
		for (n1 = 0; n1 <= topology[layer]; n1++) {
			if (n1==0 && layer==numlayers-1){
				//bias neuron on OUTPUT layer - ignore, it doesn't exist
			} else {
				xpos = leftmargin + n1 * xgap;
				ctx.strokeStyle='LightGrey';
				if (n1==0){ //bias on any layer except output
					ctx.fillStyle = 'lime';
				} else {
					ctx.fillStyle = neuronColour(neurons[nx]);
					// ctx.fillStyle = 'lime';
				}
				ctx.beginPath();
				ctx.arc(xpos,ypos,6,0,2*Math.PI);
				ctx.fill();
				ctx.stroke();
				ctx.font = "12px Arial";
				ctx.fillStyle = '#888';
				ctx.fillText( neurons[nx].toFixed(3), xpos + 10, ypos + 2);
				ctx.fillStyle = 'red';
				ctx.fillText( errors[nx].toFixed(3), xpos + 10, ypos + 12);
			}
			nx++;
		}
	}
　
}
