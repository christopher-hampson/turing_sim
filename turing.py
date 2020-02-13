class Configuration:

    def __init__(self,state,pos,word):
        self.state = state
        self.pos = pos
        self.word = dict(word)
        self.word[pos] = self.word.get(pos," ")
        self.min_val = min(self.word.keys())
        self.max_val = max(self.word.keys())

    def __repr__(self):
        left = "".join([self.word[i] for i in range(self.min_val,self.pos)])
        right = "".join([self.word[i] for i in range(self.pos,self.max_val+1)])
        return "({0}, {1}, {2})".format(self.state,left,right)

    def __len__(self):
        return self.max_val-self.min_val+1

    def get_position(self):
        return self.pos

    def get_state(self):
        return self.state

    def get_word(self,*pos):
        if len(pos)==0:
            return self.word
        if len(pos)==1:
            return self.word.get(pos[0]," ")

    def read(self):
        symbol = self.word.get(self.pos," ")
        if symbol == " ":
            symbol = "\\blank"
        return symbol

    def evolve(self,new_state,write,move):
        new_word = dict(self.word)
        new_word[self.pos] = write
        return Configuration(new_state,self.pos+move,new_word)




class State:

    def __init__(self,id,x,y,formatting="",label=None):
        self.label = label
        self.pos = (x,y)
        self.id = id
        self.formatting = formatting.split(",")

    def __repr__(self):
        return self.get_id()

    def is_initial(self):
        return "initial" in self.formatting

    def is_accepting(self):
        return "accepting" in self.formatting

    def is_rejecting(self):
        return "rejecting" in self.formatting

    def is_final(self):
        return self.is_accepting() or self.is_rejecting()

    def get_id(self):
        return self.id

    def get_label(self):
        if self.label:
            return self.label
        else:
            return self.get_id()

    def get_pos(self):
        return self.pos

    def to_TikZ(self,attributes=""):
        attributes = self.formatting + attributes.split(",")
        return "\\node[{0}] at ({1},{2}) ({3}) {{{4}}};".format(",".join(attributes),self.pos[0],self.pos[1],self.id,self.get_label())

class Transition:

    def __init__(self,current,read,new,write,move,formatting="",label_pos="",label=""):
        self.current = current
        self.read = read
        self.new = new
        self.write = write
        move_dict = {"1":1, "0":0, "-1":-1, "-":0, "\\rightarrow":1, "\\leftarrow":-1,1:1,0:0,-1:-1}
        self.move = move_dict.get(move)
        self.formatting = formatting.split(",")
        self.label_pos = label_pos.split(",") + ["text=black"]
        self.label = label

    def __repr__(self):
        return "({0},{1};{2},{3},{4})".format(self.get_current(),self.get_read(),self.get_new(),self.get_write(),self.get_move())

    def get_label(self):
        moves = {1:"\\rightarrow",0:"-",-1:"\\leftarrow"}
        return self.label
        return "${0}/{1}/{2}$".format(self.read,self.write,moves[self.move])

    def get_current(self):
        return self.current

    def get_new(self):
        return self.new

    def get_write(self):
        return self.write

    def get_read(self):
        return self.read

    def get_move(self):
        return self.move

    def to_TikZ(self,attributes= "",label=""):
        attributes = self.formatting + attributes.split(",")
        label_attributes = self.label_pos + label.split(",")
        return "\\draw[{0}] ({1}) to node[{2}] {{{3}}} ({4});".format(",".join(attributes),self.current,",".join(label_attributes),self.get_label(),self.new)






class Turing:

    def __init__(self,diagram=None):
        self.type = "Turing"
        self.states = []
        self.initial = None
        self.trans = []
        self.accept = []
        self.reject = []
        self.halt = []
        self.word = {}
        self.diagram = diagram
        self.other_TikZ = []
        self.DFA = False

    def __str__(self):
        return "States: {0}\nInitial: {1}\nAccept : {2}\nReject: {3}\nTransitions: {4}".format(self.states,self.initial,self.accept,self.reject,self.trans)

    def add_states(self,*states):
        ## Load states for TM
        if len(states)==0:
            raise Exception("No states gives. Currently state(s) are: {0}".format(self.states))

        for s in states:
            if s in self.states:
                raise Exception("State {0} already exists".format(s))
            self.states.append(s)
                
    def get_state_ids(self):
        return [S.get_id() for S in self.states]


    def add_transition(self,trans):
        ## Load a single transitions for TM
        if str(trans.get_current()) not in self.get_state_ids():
            raise Exception("State '{0}' does not exist.".format(trans.get_current()))

        if str(trans.get_new()) not in self.get_state_ids():
            raise Exception("State '{0}' does not exist.".format(trans.get_new()))

        if trans.get_move() not in [-1,0,1]:
            raise Exception("Move must be an integer from among: -1,0,1. Got {0}".format(trans.get_move()))

        self.trans.append(trans)


    def add_transitions(self,*transitions):
        ## Load transitions for TM
        if len(transitions)==0:
            raise Exception("No transitions gives. Currently Transitions are: {0}".format(self.trans))

        for trans in transitions:
            self.add_transition(trans)

        print self.trans

    def add_initial(self,initial):
        ## Load/change initial state for TM
        if initial not in self.states:
            raise Exception("State {0} does not yet exist".format(initial))
        self.initial = initial

    def add_accept(self,*accept):
        ## Load accept states for TM
        if len(accept)==0:
            raise Exception("No states gives. Currently Accept state(s) are: {0}".format(self.accept))

        for s in accept:
            if s not in self.states:
                raise Exception("State '{0}' does not yet exist".format(s))
            self.accept.append(s)

    def add_reject(self,*reject):
        ## Load reject states for TM
        if len(reject)==0:
            raise Exception("No states gives. Currently Reject state(s) are: {0}".format(self.reject))

        for s in reject:
            if s not in self.states:
                raise Exception("State '{0}' does not yet exist".format(s))
            self.reject.append(s)

    def add_halt(self,*halt):
        ## Load halting states for TM
        if len(halt)==0:
            raise Exception("No states gives. Currently Halting state(s) are: {0}".format(self.halt))

        for s in halt:
            if s not in self.states:
                raise Exception("State '{0}' does not yet exist".format(s))
            self.halt.append(s)

    def load_word(self,word,padding=4):
        ## Load input word onto the tape
        self.word = dict([(i,word[i]) for i in range(len(word))])

    def get_state_by_id(self,id):
        for state in self.states:
            if state.get_id() == id:
                return state
        raise Exception("No state with that id exists!")


    def get_successors(self,current_state,read):
        successors = []
        for T in self.trans:
            if T.get_current() == current_state and T.get_read() == read:
                # successors.append((self.get_state_by_id(T.get_new()),T.get_write(),T.get_move()))
                successors.append(T)
        return successors

    def run(self,n,current_run=None,transition_sequence=[]):
        ## Generator that yields runs of length n

        if current_run==None:
            current_run = [Configuration(self.initial,0,self.word)]
            transition_sequence = []

        if n==0: 
            yield current_run, transition_sequence

        config = current_run[-1]

    
        current_state = config.get_state()
        read = config.read()

        if current_state in self.halt:
            yield current_run, transition_sequence

        successors = self.get_successors(current_state,read)

        if not successors:
            if len(self.reject)>0:
                reject_state = self.reject[0]
                reject_config = config.evolve(reject_state,read,0)
                current_run = current_run+[reject_config]
                transition_sequence.append(None)
            yield current_run, transition_sequence

        for T in successors:

            new_state = T.get_new()
            write = T.get_write()
            move = T.get_move()

            next_config = config.evolve(new_state,write,move)

            G = self.run(n-1,current_run+[next_config],transition_sequence = transition_sequence + [T])
            yield next(G)


    def build_LaTeX_tape(self,run,seq,output='out',n=8):
        import os, re 

        ## read tape template and split into sections ##
        f = open("template.tex","r")
        s = f.read()

        m = re.search('([\s\S]*)(\\\\begin{tikzpicture}[\s\S]*\\\\end{tikzpicture})([\s\S]*)', s)
        head = m.group(1)
        pic = m.group(2)
        foot = m.group(3)
        f.close()


        ## open tape output for writing ##
        output_file = output + "_tape"
        f = open("{0}.tex".format(output_file), "w")
        f.write(head)
        for config in run:

            f.write("\\begin{tikzpicture}")

            f.write("\\draw[draw=none] (-0.75,-0.75) rectangle ({0},2);".format(len(config.get_word())+1.5))

            # draw tape (non-active)
            for i in range(len(config)):
                if config.get_position() == i:
                    pass    # active cell added later
                else:
                    f.write("\\node[tape] (w{0}) at ({1},0) {{${2}$}};\n".format(i,1.0*i,config.get_word(i)))

            # draw end tape cell
            f.write("\\node[endtape] (w{0}) at ({1},0) {{$\\dots$}};\n".format(len(config),len(config)+0.25))

            # draw tape (active cell)
            i = config.get_position()
            f.write("\\node[tape,active] (w{0}) at ({1},0) {{${2}$}};\n".format(i,1.0*i,config.get_word(i)))


            # draw tape head
            if config.get_state() in self.accept:
                f.write("\\node[head,accept] at ({0},0) {{{1}}};\n".format(config.get_position(),config.get_state().get_label()))
            elif config.get_state() in self.reject:
                f.write("\\node[head,reject] at ({0},0) {{{1}}};\n".format(config.get_position(),config.get_state().get_label()))
            else:
                f.write("\\node[head] at ({0},0) {{{1}}};\n".format(config.get_position(),config.get_state().get_label()))

            f.write("\\end{tikzpicture}\n")
        f.write(foot)
        f.close()

        # compile LaTeX
        os.system("pdflatex {0}.tex >/dev/null".format(output_file))

        print "TEST"

        # remove auxiliary files
        os.system("rm {0}.log".format(output_file))
        os.system("rm {0}.aux".format(output_file))

        print "Successfully compiled"

    def build_LaTeX_sim(self,run,seq,output='out',n=8):
        import os, re 

        ## read tape template and split into sections ##
        f = open(self.diagram,"r")
        s = f.read()

        m = re.search('([\s\S]*)\\\\begin{tikzpicture}([\s\S]*)\\\\end{tikzpicture}([\s\S]*)', s)
        head = m.group(1)
        pic = m.group(2)
        foot = m.group(3)
        f.close()


        ## open output for writing ##
        output_file = output + "_sim"
        f = open("{0}.tex".format(output_file), "w")
        f.write(head)
        for i in range(len(run)):
            config = run[i]

            if i>0:
                last_trans = seq[i-1]
            else:
                last_trans = None

            f.write("\\begin{tikzpicture}\n")
            f.write("\n".join(self.other_TikZ))

            for state in self.states:
                if state == config.get_state():
                    f.write(state.to_TikZ(attributes="active"))
                else:
                    f.write(state.to_TikZ())
                f.write("\n")

            for edge in self.trans:
                if edge == last_trans:
                    f.write(edge.to_TikZ(attributes="last edge"))
                else:
                    f.write(edge.to_TikZ())
                f.write("\n")

            state = config.get_state()
            pos = state.get_pos()


            #f.write(config.to_latex())
            f.write("\\end{tikzpicture}\n")
        f.write(foot)
        f.close()

        # compile LaTeX
        os.system("pdflatex {0}.tex >/dev/null".format(output_file))

        # remove auxiliary files
        os.system("rm {0}.log".format(output_file))
        os.system("rm {0}.aux".format(output_file))

        print "Successfully compiled"


    def build_LaTeX(self,n=8):
        output_file = self.diagram[:-4]

        # ## get TM run ##
        gen = self.run(n)
        run, seq = next(gen)

        ## Generate tape ##
        self.build_LaTeX_tape(run,seq,output=output_file,n=20)

        ## Generate sim ##
        try:
            self.build_LaTeX_sim(run,seq,output=output_file,n=20)
        except:
            print "Failed to generate simulation"

        return None



    def parse_LaTeX(self):
        import os, re 

        filename = self.diagram
        ## read tempalte and split into sections ##
        f = open(filename,"r")
        s = f.read()

        m = re.search('([\s\S]*)\\\\begin{tikzpicture}([\s\S]*)\\\\end{tikzpicture}([\s\S]*)', s)
        head = m.group(1)
        pic = m.group(2)
        foot = m.group(3)
        f.close()


        dict_of_states = {}
        list_of_edges = []
        other_TikZ = []
        for line in pic.split("\n"):
            ## Parse states nodes
            m = re.match('\\\\node\[([\s\S]*?)\]\s*at\s*\(([\s\S]*?),([\s\S]*?)\)\s*\(([\S]*?)\)\s*\{([\s\S]*?)\};', line)
            if m:
                formatting = m.group(1)
                x = m.group(2)
                y = m.group(3)
                id = m.group(4)
                label = m.group(5)
                dict_of_states[id] = State(id,x,y,formatting=formatting,label=label)
                continue

            m = re.match('\\\\node\[([\s\S]*?)\]\s*\(([\S]*?)\)\s*at\s*\(([\s\S]*?),([\s\S]*?)\)\s*\{([\s\S]*?)\};', line)
            if m:
                formatting = m.group(1)
                x = m.group(2)
                y = m.group(3)
                id = m.group(4)
                label = m.group(5)
                dict_of_states[id] = State(id,x,y,formatting=formatting,label=label)
                continue


            ## Parse egdes
            m = re.match('\\\\draw\[([\s\S]*?)\]\s*\(([\S]*?)\)\s*to node(\[([\s\S]*?)\])?\s*\{([\s\S]*?)\}\s*\(([\S]*?)\);', line)
            if m:
                formatting = m.group(1)
                current_state = dict_of_states.get(m.group(2))
                new_state = dict_of_states.get(m.group(6))
                label_pos = m.group(4)
                label = m.group(5).replace("$","")
                if self.DFA:
                    read, write, move = label, label, 1
                else:
                    read, write, move = label.split("/")
                T = Transition(current_state,read,new_state,write,move,formatting=formatting,label_pos=label_pos,label=m.group(5))
                list_of_edges.append(T)
                continue


            ## save line
            other_TikZ.append(line)

        ## add to machine
        for i in dict_of_states:
            state = dict_of_states[i]
            print state
            self.add_states(state)
            if state.is_initial():
                self.add_initial(state)

            if state.is_accepting():
                self.add_accept(state)
                if not self.DFA: 
                    self.add_halt(state)

            if state.is_rejecting():
                self.add_reject(state)
                if not self.DFA:
                    self.add_halt(state)

        for edge in list_of_edges:
            self.add_transition(edge)

        self.other_TikZ = other_TikZ

        #return dict_of_states,list_of_edges, other_TikZ





class DFA (Turing):

    def __init__(self,diagram=None):
        self.states = []
        self.initial = None
        self.trans = []
        self.accept = []
        self.reject = []
        self.halt = []
        self.word = {}
        self.diagram = diagram
        self.other_TikZ = []
        self.DFA = True




## DFA (mod 3) example
A = DFA(diagram='NFA.tex')
A.parse_LaTeX()


print A


A.load_word('abbaba  ')
gen = A.run(25)

run, seq = next(gen)

print run, seq


A.build_LaTeX_tape(run,seq,output='NFA',n=20)
A.build_LaTeX_sim(run,seq,output='NFA',n=20)



## DFA (mod 3) example
A = DFA(diagram='DFAmod3.tex')
A.parse_LaTeX()


print A


A.load_word('1001101010  ')
gen = A.run(25)

run, seq = next(gen)

print run, seq


A.build_LaTeX_tape(run,seq,output='DFAmod3',n=20)
A.build_LaTeX_sim(run,seq,output='DFAmod3',n=20)


filename = 'TM_anbn.tex'
M = Turing(diagram=filename)
M.parse_LaTeX()

print M


M.load_word('aaabb    ')

M.build_LaTeX(35)


filename = 'example2.tex'
## anbn example
M = Turing(diagram=filename)
M.parse_LaTeX()

print M


M.load_word('RRLRLL   ')

M.build_LaTeX(35)

quit()

init = State('init',0,0,label='q_{\\mathsf{init}}')
acc = State('accept',1.2,0,label='q_{\\mathsf{accept}}')
rej = State('reject',0,-2,label='q_{\\mathsf{reject}}')

q1 = State('q1',2,0.8,label='q_1')
q2 = State('q2',3,0,label='q_2')
q3 = State('q3',2,-0.8,label='q_3')

M.add_states(init,q1,q2,q3,acc,rej)
M.add_initial(init)
M.add_accept(acc)
M.add_reject(rej)
M.add_halt(acc,rej)
M.add_transition(Transition(init,'a',q1,' ',1))
M.add_transition(init,'b',rej,' ',0)
M.add_transition(init,' ',acc,' ',0)
M.add_transition(q1,'a',q1,'a',1)
M.add_transition(q1,'b',q1,'b',1)
M.add_transition(q1,' ',q2,' ',-1)
M.add_transition(q2,'a',rej,' ',0)
M.add_transition(q2,'b',q3,' ',-1)
M.add_transition(q2,' ',rej,' ',0)
M.add_transition(q3,'a',q3,'a',-1)
M.add_transition(q3,'b',q3,'b',-1)
M.add_transition(q3,' ',init,' ',1)

print M


M.load_word('aaaabb  ')


M.build_LaTeX(30)
quit()


