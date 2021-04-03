from typing import Any, Optional

import matlab.engine


class MATLABHandler:
    def __init__(self, engine_name: Optional[str] = None):
        if engine_name:
            try:
                self.engine = matlab.engine.connect_matlab(engine_name)
            except:
                self.engine = matlab.engine.start_matlab()
        else:
            self.engine = matlab.engine.start_matlab()

    def quit(self):
        self.engine.quit()

    def eval(self, instr: str, nargout: int = 0) -> Any:
        return self.engine.eval(instr, nargout=nargout)

    def eval_args(
        self, func: str, arg: str, nargout: int = 0, include_semicolon: bool = True
    ) -> Any:
        string = self.__class__.make_eval_string(func, arg, include_semicolon)
        return self.eval(string, nargout=nargout)

    @staticmethod
    def make_eval_string(func: str, arg: str, include_semicolon: bool = True) -> str:
        semic = ";" if include_semicolon else ""
        return f"{func}({arg}){semic}"
