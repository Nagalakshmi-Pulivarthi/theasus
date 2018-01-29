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
                $('#predict_result').append("<tr><td>aoc_score_test</td><td>"+data.aoc_score_test+"</td></tr>");
                $('#predict_result').append("<tr><td>logit_roc_auc</td><td>"+data.logit_roc_auc+"</td></tr>");
                $('#predict_result').append("<tr><td>logit_roc_auc</td><td>"+data.logit_roc_auc+"</td></tr>");

            },
        });
    });
});