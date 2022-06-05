/******************************************************** 
 * Ajax통신 기능
********************************************************/
var AnimalRequest = function( url, success) {
	if( url == null || url == "undefined") {
		alert("Check URL.");
		return false;
	}

	if( success == null || success == "undefined" ) {
		alert("Check success function.");
		return false;
	}

	var ajax_opt = {
			url: url,
			type: "POST",
			contentType: "application/json; charset=utf-8",
            dataType: "json",
			//data: JSON.stringify(params),
			beforeSend:function(){},
			complete:function(){},
			error: function( request, status, e) {
				console.log("[오류내용]:" + request.responseText + "\n" + "status:" + status + "\n" + "error:" + e);
		    },
			success: success,
	};

	$.ajax( ajax_opt);
}

var getPrediction = function( url, params, success) {
	if( url == null || url == "undefined") {
		alert("Check URL.");
		return false;
	}

	if( success == null || success == "undefined" ) {
		alert("Check success function.");
		return false;
	}

	var ajax_opt = {
			url: url,
			type: "POST",
			contentType: "application/json; charset=utf-8",
            dataType: "json",
			data: JSON.stringify(params),
			beforeSend:function(){},
			complete:function(){},
			error: function( request, status, e) {
				console.log("[오류내용]:" + request.responseText + "\n" + "status:" + status + "\n" + "error:" + e);
		    },
			success: success,
	};

	$.ajax( ajax_opt);
}
