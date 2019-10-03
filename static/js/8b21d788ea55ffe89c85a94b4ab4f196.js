(function (root, factory) {if (typeof define === 'function' && define.amd) {define(['exports', 'echarts'], factory);} else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {factory(exports, require('echarts'));} else {factory({}, root.echarts);}}(this, function (exports, echarts) {var log = function (msg) {if (typeof console !== 'undefined') {console && console.error && console.error(msg);}};if (!echarts) {log('ECharts is not Loaded');return;}if (!echarts.registerMap) {log('ECharts Map is not loaded');return;}echarts.registerMap('恩阳区', {"type":"FeatureCollection","features":[{"type":"Feature","id":"511903","properties":{"name":"恩阳区","cp":[106.654386,31.787186],"childNum":2},"geometry":{"type":"MultiPolygon","coordinates":[["@@@ABAA@@@A@@@@BBB"],["@@@ADC@E@A@CBCLODA@AACAAACEA@CCKAAC@EAACAABABCBA@AA@E@A@ACAEBGCAG@GAEAEAEAEBGBIDA@CBOJCBE@GAC@A@GBEB@@CDADAD@@BDDBBFBDADEF@B@BBBBB@@@B@@BD@B@@@BABA@E@@BAHADADABCBCBC@ABAD@BABIFABGD@BABABC@C@A@AAAACBABADC@CBE@ABA@CABAAAAE@CA@AAA@CCA@E@CAABCDCBA@A@AAA@A@C@A@AB@BAB@B@DAD@BED@BAD@D@BABA@@@A@ECC@CAABE@ABA@ABABA@EBAACBABABADBD@BABA@CBA@AB@BBBFFBB@F@H@FABADABBD@DDBBB@DAPADABABC@CBAB@B@@BBBBBB@BAFAB@BABCBCBA@ADCFAB@@A@A@A@CCA@AA@@A@E@CBEB@@@B@BA@A@ABA@@BA@@BBD@@@@AB@@AAA@@CAA@@@AA@ABAAAAA@@B@B@@@BA@CAAB@@AB@BA@@B@@@@@B@@@BBB@@@BB@AB@BABA@A@ABA@A@A@AB@@BD@B@@@B@@@B@@BB@@ABAB@@BB@@ABAB@@CBEDABED@@A@A@C@KDCBA@ABCDABA@GBCBABA@AF@@@B@BBDBBDBHBB@BB@B@B@B@BD@B@BB@DBBBBDDBBBDAB@BCDAD@BAF@BDBD@B@@A@CBABAB@FBB@DABAH@BABADCB@B@BBBBBBBD@FADBBBBBBBBBD@B@HABA@EDAD@DFHD@DBFAF@DAH@B@@B@BABCBADABABCHABGFEHABADAD@F@BADAD@@A@ABA@@B@@BBDBB@@BEDABADABAF@B@DCDD@D@HDB@DBBDAFBFAD@BBB@BDDBFFHFDBD@BABCBCD@@ABABCJAB@D@D@@@D@B@BA@@BBFBFBB@DAD@FCH@B@B@BBDDFHDJDDBFFBB@B@BADABC@A@C@A@E@C@A@ABABAD@B@B@BBBDBD@J@DBBBDBBBDB@BBBBD@F@@@@BB@J@D@@BBDBFJBBBBBB@@DBFFFJBDDDDBFBF@BABCDCF@B@DBBF@DCND@DADGBCBCDE@@DCFCDAH@DBBADDFDB@F@F@DAHBFDB@FBNB@@FDD@DA@@LC@@B@@B@B@@AD@D@B@@BBLKLOBCDCBAFEDADADCB@BAPEFCB@JKBGBIBABADCB@BAB@AA@@@@@@AA@@@@B@@@B@B@@A@@CGCC@A@AB@@@B@B@@@BB@@B@@@@@BA@@@@@ACGAC@A@@BAB@HCB@BA@AAAACA@@ABAB@B@B@DBB@F@@@BAB@@A@@BC@ABCB@@AB@FAB@D@B@D@BBB@HBAC@A@C@A@C@C@ABABA@@BAF@D@DABABA@CBA@C@A@C@ABABADA@A@AB@B@BBBBDDBD@BDBBBB@B@BABA@@BC@@@A@ADCBC@CBA@@BAF@F@DAFBBBDBDBD@F@D@DBB@B@DABA@A@AB@@AB@@A@AAA@A@CBCDABABAD@D@B@@BDBB@@@B@B@@A@A@A@@@A@E@GBI@A@@AAEEACAAAA@A@A@AAA@@A@C@A@A@@@AA@@BA@@BAAA@@BA@@BA@AA@A@A@A@ADABA@A@@ABADE@@BADABABAB@BAAA@C@A@C@C@ABAB@B@B@@BBB@BBD@@DBD@B@B@D@D@DCBAFE@A@C@A@AB@BAB@@AB@B@DBDDBB@BB@B@D@D@FAB@@ABAFOBCBAAABEAA@ACAAAAAAAC@AA@A@A@ABADAB@BABA@@AC@CAAAA@AA@@AAA@A@AB@BA@@BA@A@ABAB@LAB@B@@BBDB@B@B@BAB@B@@@B@BB@BBBB@B@B@BA@A@A@A@A@A@@@AB@B@B@B@BDDDB@BBB@B@D@@@BAB@BAB@B@@@B@@BB@@BBBBB@BB@BBBA@@@I@ABA@@AA@AAAACA@AA@C@@BA@@BCB@@A@AA@AAAAAAA@@@AA@AA@CAA@AA@@A@AAA@A@AAJG@@HEHCB@BA@A@C@@@ABABABA@@@AGDABA@@AAABABAB@DCD@D@B@BABADCF@B@D@BBB@B@B@D@BA@A@@@AB@BAJEDCEGGGEACC@@@ABA@@BA@A@@AA@A@@@AFGB@B@BBBBBFDB@@D@@@B@@A@@AAA@CC@AB@B@D@BA@@@ACC@@AA@@@A@@@AA@@@AAABA@@@ABA@@@A@A@@@A@ABA@@@A@AA@@A@A@A@ABA@AA@ACC@AAA@@AA@@@@@A@A@ABABAB@BC@@@E@C@G@@@A@A@CAEKGAA@C@KCG@A@C@AB@B@BAD@@A@@AAC@C@A@C@ABCBCBABA@@AAAAA@A@ACBABA@A@CAA@C@E@ABCB@@@BAF@DABABCDAD@FAB@BBBB@BBB@BB@BAB@@A@AAA@C@@BCBA@@@AAAACBABA@CBC@CBABABA@@A@A@AAA@@C@C@A@CDCB@BC@AB@BA@ABA@A@AA@A@ABABA@A@C@ABADABAB@BA@AACAACAACAAAA@CBA@A@C@EACACA@@ABABCBEDKIAEAC@EACAEAAAAAAC@AAA@AA@G@CBCBCF@B@D@BA@A@A@E@CAA@A@AD@B@DADADEBAD@D@BB@D@LCD@D@BB@B@BABABCDGBGAEAAABAAA@AC@C@@@AAA@A@AB@B@@@B@@@@ABA@@@@BC@A@ABCDAB@@CACAAA@C@A@CAAAAAA@AAC@AGECCAC@C@A@@BCBA"]],"encodeOffsets":[[[109143,32668]],[[109194,32322]]]}}],"UTF8Encoding":true});}));