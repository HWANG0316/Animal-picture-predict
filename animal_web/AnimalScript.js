	        var canvas = document.getElementById("myCanvas");
	        var myContext = canvas.getContext("2d");

	        var now_x; var now_y;  var hint_time; var age; var com;

	        //3차원 벡터. 최종적으로 서버에 넘겨지는.
	        var stroke=[[[],[],[]]]; 

	        //2차원 벡터. 한 획을 담는 tmp stroke vector.
	        var stroke_tmp=[[],[],[]];

	        //next누를 때마다 전송되는 age, company정보 담을 array.
	        var user_info=[[],[]];

	        var stopped = true;

	        //현재 목표그림 동물
	    	var play="초기값입니다.";
	    	var now_idx=0;

	    	//동물 한 개당 30초 시간제한
	    	var time_limit=30;
	    	var sec="";

	    	function getPlay(){

	    		return play;

	    	}
	    	function setPlay(p){

	    		play=p;

	    	}

	    	function timer(){

	    		time_limit--;
	    		document.getElementById("timer").innerText="시간제한: "+time_limit;

	    		if(time_limit==0){

	    			time_limit=30;
	    			alert("시간이 완료되었습니다.");

	    			//다음 그림으로 넘어가기 
	    			next(); 

	    		}

	    	}

	        function start() { //canvas에 마우스 움직임 시작 

	            e = window.event;
	            myContext.moveTo(e.clientX-110, e.clientY-210);
	            stopped = false;
	           
	        }

	        function stop() { //canvas에 마우스 움직임 멈춤

	            stopped = true;

	            //한 획 저장
	            stroke.push(stroke_tmp);

	            //현재 3차원 stroke vector를 서버로 전송
	            predict(getStroke(),getPlay());

	            //한 획 담을 배열 초기화
	            resetStroketmp();

	        }

	        function draw() { //canvas에서 마우스 움직이는 중

	            if (!stopped) {

	                e = window.event;
	                myContext.lineTo(e.clientX-110, e.clientY-210);      
	                myContext.stroke();

	                //마우스 움직임따라 stroke vector 값 생성
	                store();

	            }

	        }

	        function button_erase(){ //캔버스 초기화

	        	myContext.clearRect(0, 0, canvas.width, canvas.height);
	        	myContext.beginPath();

	        }

	        function store(){ //stroke vector 값 생성 

	        	now_x=e.clientX-110;
	            now_y=e.clientY-210;
	            hint_time=performance.now()/1000;

	            if(hint_time>10){ //10초가 지나면 힌트가 보이도록

	      			document.getElementById("hint").style.visibility="visible";

	      		}

	            stroke_tmp[0].push(now_x);
	           	stroke_tmp[1].push(now_y);
	           	stroke_tmp[2].push(hint_time);

	            //console.log("x: "+now_x+" y: "+now_y+" time: "+now_time);
	        }

	        function next(){

	        	button_erase();

	        	//현재 동물 array index 증가
	        	now_idx++;

	        	//목표 동물 setting
	        	animal_set(now_idx); 

	        	//목표, 현재 맞춘 개수 setting
	        	goal_set(now_idx);

	        	//3차원 stroke vector 초기화
	        	resetStroke();

	        	age = document.getElementById("yourage").value;
	        	com = document.getElementById("company").value;
	        	//console.log("age: "+ age + " company: " +com);
	        	
	        	user_info[0].push(age);
	        	user_info[1].push(com);

	        	console.log(user_info);
	        	//document.write(user_info);

	        }

	        function getStroke(){

	        	return stroke;

	        }

	        function resetStroke(){

	        	stroke=[[[],[],[]]];

	        }

	        function resetStroketmp(){

	        	stroke_tmp=[[],[],[]];

	        }

		    //global array: 서버에서 동물리스트를 받아 저장
		    var animal_list= [[]];

		    function init(){

		    	animal_load();

		    	 //timer start
	            clearInterval(0);
	            sec=setInterval("timer()",1000);

		    }

		    function list_reset(){

		    	animal_list=[[]];

		    }

		    function goal_set(index){ //목표, 현재 맞춘 개수 setting

		    	var now_num=document.getElementById("now");
		    	var goal_num=document.getElementById("goal");

		    	now_num.innerHTML="현재 "+index+"개";
		    	goal_num.innerHTML="목표 "+(30-index)+"개 /";

		    	if(index==29){ //0부터 29까지 그리기 완료했다면.

		    		now_num.innerText="현재 30개";
		    		goal_num.innerText="목표 0개 / ";

		    		gameover();

		    	}

		    }

		    function gameover(){

		    	var choice =confirm("게임을 종료하시겠습니까? '취소'를 누르면 재도전을 시작합니다.");

		    	if(choice){ //'확인'을 누름

		    		//console.log("게임 종료.");

		    		//현재 창 닫기.
		    		self.close();

		    		//game over 창 열기.
		    		window.open("pop.html","game over window");

		    	}

		    	else{

		    		//console.log("재도전 시작.");

		    		//array를 reset하고 server에서 새로운 list를 받아온다.
		    		list_reset();
		    		animal_load();

		    	}

		    }

		    function animal_set(index){

		    	//영어로 setting. server에 보내질 변수.
		    	setPlay(animal_list[0][index][1]);

		    	//확인용
		    	//console.log(getPlay());

		    	var animal = document.getElementById("animal");
		    	//var hint=document.getElementById("hint");

		    	//태그에는 한글로 세팅.
		    	animal.innerText=animal_list[0][index][0];
		    	//hint.innerText=now_animal.hint;

		    }
		    

		    function animal_load(){ //서버로부터 28개의 동물list를 받아온다.

		    	console.log( 'animal_load');

				//'Loaded': return value's name
		    	var success = function (loaded){ 
		    		

		    		if( loaded.success) { //서버와 연결 성공 시 

		    			for(i in loaded.data){

		    				var animal_tmp=[];
		    				animal_tmp.push(loaded.data[i].code_nm); //한글 동물이름
		    				animal_tmp.push(loaded.data[i].code_cd); //영어 동물이름

		    				animal_list[0].push(animal_tmp);
		    				
		    			}

		    			//28개를 서버에서 받아온 후 2개를 추가하여 30개로 setting.
		    			animal_list[0].push(animal_list[0][8]);
		    			animal_list[0].push(animal_list[0][18]);
		    			
		    			console.log(animal_list);

		    			//첫 동물을 목표로 setting.
		    			animal_set(0); 
		    			goal_set(0);

		    		}

		    		else {

		    			alert("animal_load: Failing to get data from server.");

		    		}
		    	};
		    	
		    	//make ajax object (connect to AnimalRequest.js)
		    	new AnimalRequest('http://ec2-13-209-65-153.ap-northeast-2.compute.amazonaws.com:8080/xr.api/xr/api/drawing/collect/list',success);

		    	
		    }

		    function animal_search(ani_eng){ //영어 동물이름을 받아 한글 동물이름을 return
		    	
		    	//console.log("영어예측동물: "+ani_eng);
		    	for(var i=0; i<animal_list[0].length; i++){

		    		//console.log("i증가확인 :"+i+"동물:"+animal_list[0][i][1]);
		    		if(animal_list[0][i][1]==ani_eng){

		    			return animal_list[0][i][0];

		    		}
		    	}

		    	//한글로 매칭되는 것이 없는 것들 예외처리.ex: snail..
		    	if(ani_eng=="circle") return "원";
		    	else if(ani_eng=="triangle") return "삼각형";
		    	else if(ani_eng=="flower") return "꽃";
		    	else if(ani_eng=="sun") return "태양";
		    	else return ani_eng; 

		    }
		    

		    function predict(strk,ani){ //서버에게 동물예측 요청

		    	//object 생성
		    	var objParam={};

		    	console.log(strk);
		    	console.log(ani);

		    	objParam.drawing=strk;
		    	objParam.playing_word=ani;

		    	if(!strk){

		    		alert("stroke vector가 없습니다.");
		    		return;

		    	} 
		    	if(!ani){

		    		alert("현재 동물 정보가 없습니다.");
		    		return;

		    	} 

		    	//'Prediction': return value's name
		    	var success = function (dd){
		    		
		    		//store the server's return values
		    		var mess; 
		    		var pred_animal;

		    		if(dd.success) { //서버와 연결 성공 시 
		    				
		    				//안 쓰임. 값 없음.
		    				mess=dd.message; 

		    				//서버에서 예측한 동물
		    				pred_animal=dd.data;

		    			//console.log(dd.message);
		    			console.log(pred_animal);

		    			//'2'는 성공 '1'은 실패
		    			if(objParam.playing_word == pred_animal ){

		    				stroke_store(2);
		    				answer(2,pred_animal);

		    			}

		    			else{

		    				stroke_store(1);
		    				answer(1,pred_animal);

		    			}
		    		}

		    		else {

		    			alert("predict: Failing to get data from server.");

		    		}
		    	};
		    	
		    	//make ajax object (connect to AnimalRequest.js)
		    	new getPrediction('http://ec2-13-209-65-153.ap-northeast-2.compute.amazonaws.com:8080/xr.api/xr/api/drawing/predict',objParam,success);
		    }

		    function answer(flag, pred){ //정답 comment setting

		    	var prediceted=document.getElementById("result_mess");

		    	var hangle_result=animal_search(pred);
		    	//console.log("한글예측결과 :"+hangle_result);

		    	if(flag==1){

		    		prediceted.innerText="혹시 ["+hangle_result+"]그림을 그리셨나요?";

		    	}

		    	else if(flag==2){

		    		prediceted.innerText="정답입니다! 완벽한 ["+hangle_result+"]그림이에요.";

		    	}
		    }

		    function stroke_store(flag){ //서버에 data 저장. 

		    	var objParam={};

		    	objParam.drawing=getStroke();
		    	objParam.playing_word=getPlay();

		    	if(flag==1){ //match success

		    		objParam.is_end=1;
		    		//console.log("flag is 1");

		    	}
		    	else if(flag==2){ //match fail

		    		objParam.is_end=2;
		    		//console.log("flag is 1");

		    	}
		    	else{
		    		console.log("error: stroke_store param is wrong");
		    	}

		    	var success = function (store_mes) 
		        {
		    		
		    		console.log(objParam);
		    		
		    		if(store_mes.success) {
		    			
		    				console.log(store_mes.data);
		    		}
		    		else {

		    			alert("store_mes: Failing to get data from server.");
		    			
		    		}
		    	};
		    	
		    	//make ajax object (connect to AnimalRequest.js)
		    	new getPrediction('http://ec2-13-209-65-153.ap-northeast-2.compute.amazonaws.com:8080/xr.api/xr/api/drawing/predict',objParam,success);

		    }