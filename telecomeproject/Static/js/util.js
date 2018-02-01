$(function() {
    $('#predict_result').hide();
    $('#upload-file-btn').click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/upload',
            data: form_data,    
            contentType: false,
            cache: false,
            processData: false,
            async: false,
            success: function(data) {
                console.log('Success!');
                $('#predict_result').html('');
                $('#predict_result').show();
                $('#predict_result').append("<tr><td>RecordsReceived</td><td>"+data.RecordsReceived+"</td></tr>");
                $('#predict_result').append("<tr><td>RecordsProcessed</td><td>"+data.RecordsProcessed+"</td></tr>");
                $('#predict_result').append("<tr><td>Predictedchurncount</td><td><a onclick='showChurnData();return false;' href='#'>"+data.Predictedchurncount+"</a></td></tr>");
            },
        });
       // alert("File Submitted for processing. Please wait for the results");
    });
});
function showChurnData(){
    $.ajax({
        type: 'GET',
        url: '/getChurnData',
        contentType: false,
        cache: false,
        processData: false,
        async: false,
        success: function(data) {
            console.log('Success!');
            $('#data_result').html('');
            $('#data_result').show();
            $('#data_result').append("<span><a class='download-file' href='/downloadChurnData'>Download</a></span>");
            var table = $.makeTable(data);
            $(table).appendTo("#data_result");
            return;
        }
    });
}

$.makeTable = function (mydata) {
    var table = $('<table border=1>');
    var tblHeader = "<tr>";
    for (var k in mydata[0]) tblHeader += "<th>" + k + "</th>";
    tblHeader += "</tr>";
    $(tblHeader).appendTo(table);
    $.each(mydata, function (index, value) {
        var TableRow = "<tr>";
        $.each(value, function (key, val) {
            TableRow += "<td>" + val + "</td>";
        });
        TableRow += "</tr>";
        $(table).append(TableRow);
    });
    return ($(table));
};