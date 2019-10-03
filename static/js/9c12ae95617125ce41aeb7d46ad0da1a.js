(function (root, factory) {if (typeof define === 'function' && define.amd) {define(['exports', 'echarts'], factory);} else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {factory(exports, require('echarts'));} else {factory({}, root.echarts);}}(this, function (exports, echarts) {var log = function (msg) {if (typeof console !== 'undefined') {console && console.error && console.error(msg);}};if (!echarts) {log('ECharts is not Loaded');return;}if (!echarts.registerMap) {log('ECharts Map is not loaded');return;}echarts.registerMap('靖宇县', {"type":"FeatureCollection","features":[{"type":"Feature","id":"220622","properties":{"name":"靖宇县","cp":[126.813583,42.388896],"childNum":1},"geometry":{"type":"Polygon","coordinates":["@@AB@@AB@@ABA@ABABAB@@ABA@ABA@@@A@A@@@A@A@@@A@A@AA@@A@@@A@A@ABA@A@@@A@@@A@@@A@A@A@A@@@A@@AA@@@C@A@C@@@A@AA@AA@@@BA@@BA@@@A@@AA@@AAAA@A@@@A@A@@@A@@@AA@AA@AA@A@A@A@A@A@@@A@@@AAAAA@@@A@A@A@A@EBCBA@A@@@@AA@AAAAAA@AAAAC@AAA@A@A@@BA@@@@@@C@A@A@A@A@A@@@@BABAB@BA@ABA@@@A@C@A@A@ABA@AB@@ABA@ABA@ABA@@@A@A@@@A@@@@BAB@B@@@BA@@@A@@@CACACAC@AAEAA@AAA@@@AB@@CBABC@ABA@A@@@A@A@A@@BA@@BA@A@A@A@@A@@AA@AA@@@A@A@@@A@A@A@@@A@A@@@ABA@A@@@A@@@A@A@A@@@A@@@A@A@@@@@AAA@@@A@@@@@A@@@@@A@@@A@A@A@A@A@AAA@@AAA@@A@AAA@C@AAA@A@A@ABCBC@ABA@A@C@A@A@A@A@A@A@A@A@CBA@@@A@A@A@A@AAA@@@C@@@A@A@@B@@AB@B@@@@BB@@@B@@A@@BABAB@@@B@B@B@BBB@@@BBDBD@D@B@D@BA@@BA@A@A@AB@@@B@@AB@@A@@@AB@@AB@@@@@B@@@@B@@B@@@B@@@BBB@@AB@@@@AB@@@B@@@B@@@@@@@B@@@@@B@@@@@B@@ABAB@@@B@B@@@@ABA@@@A@@AA@A@@BA@@@@BA@@@AAAA@@AA@AA@@@AA@@A@AAA@A@@@A@A@C@A@AA@@A@@@@@@A@@@B@@A@ADA@@DA@@BA@ABA@A@AB@@AB@@AB@BA@@BA@@BA@@BA@@@@BAB@@@@A@@@A@@@A@A@@@A@@B@@A@@BA@A@@@AAA@@@A@AB@@ABA@A@A@A@A@@@ABAB@@ABAB@BAB@@AB@B@@@B@B@@@@@@ABA@@@AAABA@@@A@A@A@CBCBC@@@CBA@CBA@CBA@ABA@A@ABC@@BC@ABA@@@A@BB@BBD@B@B@B@B@B@BBB@B@@BBAB@@@B@BAB@@AB@BAB@@@@@@A@@@A@@@AB@@A@@B@@A@ABA@A@A@A@A@A@A@A@A@A@C@A@A@@@A@A@A@@AC@@BA@ADA@ABADCB@@ABA@A@A@A@A@A@CBABEBEBCBC@A@CBA@CBA@CB@@@BABAB@BA@@BA@@@A@A@CBA@A@@@C@@@A@@BAD@BA@A@A@ABC@AB@B@BBB@BBB@@AB@@A@ABA@@@@@@B@B@@@BB@DBBD@B@@@@@BA@@B@BB@@B@@AB@BA@@B@BA@@DA@@B@B@B@B@B@B@@@B@B@@@B@B@BB@@B@B@B@B@@@B@B@BD@BFDB@BDBB@@B@B@BAHA@@BAB@B@BABAB@D@BBD@D@BB@@BBBDBB@BBBBBB@B@B@BBB@B@BBB@@@@@B@@@BB@@@@B@@@@@B@B@@@@BD@@@@@B@@@B@@B@@B@@@B@BCDA@@@A@A@@@@@@@B@@BB@@B@@@B@@@B@@@DA@AD@BABEBABABAB@BA@ABAB@B@BA@@BCBAB@@A@ABA@@B@B@B@B@B@B@BABAB@F@@@B@BBB@B@BBB@@@B@@@@@@@B@@@BB@@B@@BB@B@@@B@@@B@@@@AB@@AB@@CBCB@@AB@@@B@@@@BBAB@B@D@@@B@@@@@@@@AB@@@@ABA@BB@@BA@BB@@@@@@@@BB@@@@B@@@@B@@@BB@@B@@@@BBB@@BB@@@@@@B@BBBBDDBDBBB@D@B@B@BBBB@B@@@@@B@@@@@BDBBB@@@@@@BB@@B@@@@@B@@@@@B@BBB@@@BBB@B@FD@@@B@@@BB@@@@B@@@@B@@BB@BBBBBBBBDB@@D@@BB@@@@@@@@@BB@@B@B@@@@AB@@@B@@BBB@@BB@@@B@@@B@@DBBB@@@@B@BB@B@@@@B@@B@@@@B@@@@@BB@@@B@B@BBDBBDB@B@BB@B@@@B@BABAB@B@L@F@JBD@F@B@DAFCFAB@@@B@@BB@@BD@B@B@BBBBBBDB@@BB@BBBB@B@BB@@B@AB@BAB@B@@@@ABB@@BB@@BB@BB@@@BB@B@@BB@@B@B@@A@BB@@@B@@@@AB@BABA@AB@BBB@BBB@@@BBB@@@D@D@@@@@@@@@@@@B@@B@@BB@@@D@@@B@@@@@@@BB@B@@@B@@BB@@BB@@BABAB@B@B@@@@DB@@AB@@@BA@AB@@AB@@@B@B@@@BAB@BBB@B@@@@B@@BB@@@BBBBB@A@ABAB@@AB@@@B@@BB@BDBBB@@@@@BA@@@AB@@@@@B@BA@@BABA@@D@@C@@B@@@B@@@B@B@@AB@@@B@@@@@B@@@B@@BB@BB@@B@D@F@@CD@BAB@B@B@B@BBB@@@B@@ABAB@@@BAB@@@B@@ABAB@BA@@BC@ABAB@@@@BBBBBB@BDDBBBB@@B@D@B@B@D@B@BAB@B@@BD@BBBBB@@B@B@B@BBBBBBBB@BBB@@@@@B@@@B@B@B@B@BABAB@D@B@B@B@BB@BBB@@B@B@@@BA@@B@BA@@@@B@@@BBBBB@@@BBB@@BB@@@@B@B@@B@@@B@BAB@B@@AB@@AB@@B@BADDBB@BB@@AB@B@D@@@BB@@@D@B@B@B@B@@@B@D@BAB@F@BBB@BAB@B@@BB@@@BB@@B@@@DBB@@BB@BD@BB@B@@B@@AB@@BDBB@DB@@F@B@D@B@@AB@@A@BBDB@BB@B@BBBBBBBBBDB@@B@@@B@B@@B@BBBBFDBBBDDD@D@@@B@@@@B@@@@@B@B@B@@BBB@D@BBBAB@@A@@BA@@B@B@BA@@BBBBBBBBBDBB@BBD@BB@@@@@@BB@@@BB@@@@B@@@@@@B@B@@@B@@@@@BABABABA@@@@B@BA@@B@@@B@DB@@B@@@B@B@@@BBB@@@B@BBD@@B@@B@@@B@@@@@@@BB@@@@@@@B@@@@BB@BB@@B@@@B@@@@@@@B@@@@@@@@B@@BB@@@@@@B@@BB@@@B@@B@@@B@B@@B@@B@@@@@@B@@BB@B@B@B@B@B@B@@@@@BB@@BBDB@@BB@@BB@@@B@@B@@@@B@@@B@@@B@@BBB@@@BB@@@DBBB@@B@D@B@B@B@@@B@B@@@B@@@B@@@@@BB@@BB@@BB@@@B@@@@BB@@@@@@@B@B@@@@@@@B@@AB@@@B@@@B@@@@@B@@@B@@@@@BAB@B@DB@@@@B@@@B@@@@@@@@A@@@@@A@@@@@A@@@@@A@@@@@AB@D@@A@@BA@@@AB@@@BAB@@@B@@@B@@@@@BA@@@AB@@@B@@@B@B@B@@@BB@@@@@@B@@@B@@@BB@@B@@BB@BB@BB@@B@@@B@@@BB@@@B@B@@@@@@BBB@@@@B@@@@@@@@@B@@@@AB@@@@B@@@BBB@@@B@@B@@@B@@@B@@B@@B@@@B@@@B@B@B@@@@@@@B@@@@BB@@@@@B@@AB@@AB@B@@@@@BB@@@BBB@B@B@B@@@@@@@@B@@@B@@@B@@B@@@@B@@B@@B@B@@BB@@@B@B@@@@@@@B@@@@B@B@B@DADBDA@@@@BBB@@@B@BBB@@BDBB@B@B@B@D@D@@@BBDB@@BB@@B@B@DB@@B@@@B@B@B@BA@@B@@@DAB@B@B@D@B@@@B@@@BB@@@@@@B@@@B@@@B@@@B@BAB@@A@AB@B@@@B@BA@@B@@@@BB@B@@BB@@@BBB@B@B@@@@@B@@@B@@BB@B@@@B@B@@@B@B@B@@@B@@B@@@BB@@@@@B@@@B@@B@@@@@@@@@B@BAD@B@BA@@B@B@B@B@@@B@@@D@B@D@@@@BB@@B@@BB@@@BB@B@BB@@@B@B@BBB@B@BB@@@@B@@@BA@@B@BA@@BA@AB@@@@@@@@@B@@@@@@@B@@@B@@AB@B@@@B@@@@@@@BB@@@@@@B@@@B@B@@@B@@@BB@@@@B@@@@@B@@@B@BAB@B@B@B@@@B@@@B@@@@@B@@@B@B@B@B@B@@@@@B@B@B@@@@@B@B@@@B@@@BB@@@@BB@B@@@B@@BB@B@@@@B@@B@@BA@@D@D@DAD@D@@@B@B@@A@@B@B@BA@@@@B@@@B@@@B@B@@@B@@@@@B@@@@@@@BA@@B@@AB@@@@@B@@@BA@@@@B@@@BAB@@AB@@@@@BB@@@@B@@@B@@A@@B@@@BB@BD@B@B@BA@A@AB@@@BABAB@BABA@@B@@ABA@ABA@A@@B@@@@@B@@@B@B@@@@@BA@@BABA@@@@@@B@@@B@B@B@B@B@B@@@DC@@B@@@B@BB@@BBBB@@BB@@BB@B@BD@B@@@BBB@BA@BB@@BBD@BB@BB@@B@@@BB@B@@@BB@BBD@BBD@B@@@B@@AB@DAB@@@B@@@B@B@BABBD@BB@@B@@@@BB@@@BB@@@@B@@@B@@@B@B@B@B@BBBB@@BB@BBBBBB@BBB@@@B@BA@ABAB@BADA@AB@B@B@B@@@BABCB@BAB@B@BADADABAB@BA@@@@BB@@BBBBB@D@BBD@D@D@B@B@@@B@BBD@@@B@B@B@D@D@DBB@@@B@B@@@@@DBDBB@B@@@B@@C@@BABCB@BA@@BBB@@@D@DBB@D@B@BBBBB@B@B@@@DA@@BA@AA@@A@@B@@@D@BBB@@BB@B@BBA@@@AB@@@@BB@@@@@B@B@BBBBB@@@@B@D@B@BBB@@@@@B@@@@BB@@@@@@@@B@@B@@B@@@@@B@@@B@BB@@BB@@@@BB@@@B@BB@@B@@@@@B@@B@@B@@BB@@@@B@@BB@@@@B@@B@@@BAB@@BB@@@BB@@@@B@@@@AB@@@@@BB@@BDD@@BB@B@@@B@@D@@BB@BB@@BBBB@BB@BBB@@@B@B@@AB@BABABA@ABABC@A@AACAAAA@A@AAAAACC@@AA@AAA@@@A@EAE@A@A@CBE@CAABCBABCBEBABABABABA@ABA@CBC@C@E@E@AAA@G@AAC@A@A@A@A@A@ABABADCBABADCBADA@A@A@A@C@ABABADCDABAD@FABADA@ABA@CBC@ABABCBABA@A@C@CBABABAB@B@DAF@DABAD@DCDABABCBA@A@E@C@IACAMAA@A@AACAE@CCEACECCCCE@@A@@@CEACCCAAAAAACACA@AAA@AAA@AACAC@AAA@AAAACAA@A@A@C@C@A@@AACACCAACAA@@AA@C@C@E@C@E@EAA@E@CAE@AAAAAAA@A@A@A@A@ABABAB@BABABABCBEBCBABA@C@C@E@C@C@AAA@C@CAAAAAAACA@ACC@AAAAAAACCAA@AAA@CAA@AAAAAAAAA@@AAA@A@C@C@C@A@A@C@A@A@AA@@AA@@BA@@@A@AA@@@AAA@A@A@AA@@AA@@BA@A@@DA@@D@B@DAD@B@B@D@BA@A@@AA@AAAAAAAAAC@AAC@CAA@C@CAA@CAEACAAAAAAA@@@AAA@C@AAE@CAA@CAC@A@A@C@A@C@C@A@C@A@C@CAE@C@C@C@A@A@@@AB@BA@@@A@@AA@AA@A@A@A@C@A@A@C@A@AA@A@A@@@A@@BA@A@@@A@A@A@@AAAAAAAAAACAAAC@AAAA@@AA@@@A@ABA@@BC@ABCBABAB@@A@@@@@A@@@@@@@AA@@@A@@@A@A@A@A@@A@@AAAAAAAA@AA@@CAA@@ACAA@@@A@A@A@A@@AA@@@@CA@@C@A@C@@AA@@@AAAAAAAAAAAAA@A@@@CAA@AA@@AA@@AA@@@AB@@@@ABAB@DAB@FADAB@@@@A@@@A@ABA@A@A@AB@@@BABAD@BADABAD@@AB@@@BA@A@A@@@A@A@AB@BABABCBAB@@A@@AAAACAAAAAAA@@AA@@AAAAAAAA@AAA@A@A@@BC@ABA@AB@BABAD@@@BA@@B@@A@A@AAA@A@AA@@AAAA@AAAAAAAAAA@AA@AA@@CAA@A@A@AA@@AAA@@AAA@AAA@C@AAA@C@A@CACCCACAC@AECCCAC@@AAAAAA@@@ABA@@B@D@B@B@B@B@DBD@B@D@B@BAD@BABA@A@@BA@@@AAA@@AAAAA@@AA@CAAAA@AAAAAA@@@A@A@@@AB@BAB@DAB@B@B@BBB@B@@AB@BAB@@ABA@A@@@A@CAAAACAACAA@E@ABCDAF@HAHAH@DAB@BABAHCDAF@JADAD@B@D@@AB@@AAA@CCCAA@AAA@C@AB@DAD@H@J@D@B@DBHDBBFHBDDBBBBBD@B@BABAFABADA@AB@B@FADBF@DBD@DBDBBB@BBBAB@B@BCBABCDABADCD@B@B@@@B@@@BB@@BB@BBD@BBB@B@D@DAB@B@@A@@BC@@BABADABA@ABAB@@AB@@@BBF@FBHDDBB@DBB@B@B@BAB@BAB@BCBA@@BA@@DABAB@B@B@B@B@BBBBBBB@DBBBBBB@@BB@@BB@BB@BB@BB@@B@BBFBBBB@BBB@@@B@B@B@B@B@B@B@BBB@B@BBBBB@@BBB@@AB@@@BAB@B@BAB@@@BB@@D@B@@BBBBB@BBBB@@B@B@@@B@BAB@B@BABABABA@A@@@@DABADA@ADAB@@AD@@ADADA@@DAB@@@@AB@@A@@@A@@@AB@@@@A@@@@@AB@@A@@B@AAA@@@@@A@@ABADCBADB@A@@@ABABABABABABA@ADC@@BA@A@AB@@A@A@A@AA@@A@AA@AACAAAAACAA@ACA@@AAA@AAAAA@AAA@AAA@@@A@@@@@@BCBABABA@AB@@AB@@AB@BA@@B@DAB@DABAB@BAB@@A@@BA@@@A@ABC@@@AA@@@@AA@@@A@AA@@A@A@A@ABA@C@A@CBA@A@A@A@@@@AA@@@@AA@@AB@@A@@@ABA@AB@@@B@@@B@B@B@B@B@BB@@B@BB@@B@@AB@@@@A@@@@@A@@AA@@@AAAA@@@AA@AAA@@AA@AAAA@@AA@@A@@A@@@@A@@@AA@@A@A@@@A@@B@@AB@B@B@@@B@@@@AB@@@BA@@@@@A@@A@@AAAA@CA@@CCEC@AA@AAA@AACAA@CAEACAA@@@A@AAC@@@A@A@@AA@@@AAA@A@A@A@AAA@A@A@AAA@C@@@AAA@@A@@@A@@@A@ABAB@BADCB@BA@AB@DA@ABA@@BA@A@@@AAA@A@AAC@AAA@C@@@A@AA@@@AAAA@AAA@C@A@AAA@@@@A@CAA@A@AAA@@A@A@AAA@@@A@@@@AA@AAA@@@@A@C@A@A@A@A@A@A@AAAAA@AAA@C@A@C@A@A@@@A@@@ABA@A@A@C@@@ABAB@B@BA@@@A@C@A@@@A@@@A@@AA@AAC@CAC@AA@@A@A@AAC@E@A@ABCBA@@BAAE@E@EAAACAA@CAE@AAA@A@ABA@@@CA@@CAA@C@AAC@A@A@@@A@A@ABA@A@A@@@@@AA@@@A@@AAA@A@AAA@@AAAA@@AA@A@A@A@@@@AAA@@@A@@@AA@@A@C@@@A@A@ABAB@@A@@@@A@A@A@ABC@A@CAA@AA@AAAA@AA@@AAAA@AAAAA@@AA@ABA@@@@AA@AAAA@A@A@AAA@@@A@A@C@A@A@A@CB@@A@A@C@@@A@@BA@AB@BABABA@@@A@@@A@A@AAA@@AA@A@@@@AAA@@@@@@A@@BA@ABAB@BA@CB@@A@@B@BADADABADABA@A@@BA@ABA@@BA@A@A@C@A@A@A@A@ABC@ABABC@ABAB@@A@@@ABA@AB@@A@A@@AA@AAA@@@@@A@@@A@A@@@@A@@@@@@@@@A@ABA@A@@BA@AD@@AB@BABA@AB@BA@@@A@@@AA@AACAA@@ACAAAAAA@A@AAA@AA@@@AB@@A@@@A@ABA@@BAB@@AB@@@@@@@BA@@@A@@@@@AA@@A@@A@@AA@@A@@BA@AA@AAAAAAA@AA@@AA@A@@BC@@@CA@@AA@AAAA@AA@BA@A@A@@A@AA@@AA@@@A@A@@A@AAAA@@AA@A@AA@@@A@CAAA@@A@@@A@A@ABC@A@AB@@AAA@A@AAAAA@@@A@AB@@AB@@@BA@@BC@CBCBA@A@A@CAA@AAA@A@A@ABA@@B@@@B@B@@@B@@A@CBCBA@A@ABA@@BA@@BA@A@AB@@EBA@@@A@A@C@ABA@@@A@@B@@@BA@@BA@@BA@@BABAB@BAB@BABABAB@@AB@@@B@BA@@@@BA@@BC@AB@BA@"],"encodeOffsets":[[129957,43153]]}}],"UTF8Encoding":true});}));