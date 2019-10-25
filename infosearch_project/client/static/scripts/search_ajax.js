function search_ajax() {
    $.ajax({
        type: "POST",
        url: "/search",
        data: $('form').serialize(),

        beforeSend: function() {
            $('#search_results').html('<h4>Ищем...</h4>')
        },

        success: function(response) {
            var parsed = $.parseJSON(response)

            $('#search_results').empty();

            if (parsed['error']) {
                $('#search_results').append(`<h3>Ошибочка на сервере: ${parsed['error']}</h3>`);
            } else if (parsed['search_result'].length === 0) {
                $('#search_results').append(
                    `<h4>Кажется, что релевантных вещей по запросу "${parsed['query']}" в корпусе нет ¯\\_(ツ)_/¯</h4>`
                 );
            } else {
                var root = $('#search_results')
                root.append(
                    `<h4>Вот что нашлось по запросу "${parsed['query']}" (искали ${parsed['elapsed_time']} c):</h4>`
                );

                for (var i = 0; i < parsed['search_result'].length; i++) {
                    root.append(`<div align="left"><span>${[i + 1]}) </span>${parsed['search_result'][i]}</div>`);
                }
            }
        },
        error: function(error) {
            $('#search_results').html('<h4>Проверь сеть и сервер - ошибка при выполнении запроса:(</h4>');
        }
    });
}
