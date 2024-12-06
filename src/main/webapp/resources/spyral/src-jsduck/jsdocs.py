import argparse, sys, os, re
from re import finditer

def parse_file(fullfilename):
    output = "\n\n// *** Documentation extracted from: "+fullfilename+" ***\n\n"
    commentsRe = re.compile(r'/\*\*(.*?)\*/[ \t]*(\r\n|\r|\n)(.*?)(\r\n|\r|\n)', re.S)
    classCodeRe = re.compile(r'class\s+(\w+)')
    classCommentsRe = re.compile(r'@class(\s+\w+)*')
    memberOfRe = re.compile(r'@memberof\s+([\w\.]+)')
    defaultsRe = re.compile(r'=\s*(\{\s*\}|undefined)')
    functionCodeRe = re.compile(r'^\s*(\w+)\s*\(')
    typeDefRe = re.compile(r'@typedef\s+\{(\w+)\}\s+(\w+)(.*)', re.S | re.M)
    typeDefs = {}
    paramRe = re.compile(r'@param\s+\{(\w+)\}\s+\[?(\w+)\]?\s*(\w*)')
    borrowsRe = re.compile(r'@borrows\s+([\w\.]+)\sas\s([\w\.]+)')
    with open(fullfilename) as f:
        contents = f.read()
        for match in finditer(commentsRe, contents):
            comments = match.group(1)
            code = match.group(3)
            if re.search(r'\w', code):
                code += "}"
            else:
                code = ""
            if "@see" in comments:
                # match see links that are on their own line and remove them (since they are links to the API)
                comments = re.sub(r'\s*\*\s+@see\s+\{@link.*?\}', "", comments)
            if "@link" in comments:
                # convert external links to <a href>
                comments = re.sub(r'\{@link\s+(http.*?)\}', r'<a href="\1" target="_blank">\1</a>', comments)
            if "@typedef" in comments:
                typeDefMatch = typeDefRe.search(comments)
                typeDefType = typeDefMatch.group(1)
                typeDefName = typeDefMatch.group(2)
                typeDefProps = typeDefMatch.group(3)
                typeDefProps = typeDefProps.replace("@property", "@param")
                typeDefProps = re.sub(r'\}\s+', '} '+typeDefName+'.', typeDefProps)
                typeDefs[typeDefName] = {
                    'name': typeDefName,
                    'type': typeDefType,
                    'props': typeDefProps
                }
                continue
            if "@param" in comments:
                for paramMatch in finditer(paramRe, comments):
                    paramType = paramMatch.group(1)
                    paramName = paramMatch.group(2)
                    typeDef = typeDefs.get(paramType)
                    if typeDef is not None:
                        comments = comments.replace(paramType, typeDef['type'])
                        typeDefProps = typeDef['props']
                        typeDefProps = typeDefProps.replace(paramType+'.', paramName+'.')
                        comments += typeDefProps
            if "static" in code:
                comments += "* @static\n"
                code = code.replace("static", "")
            if "@namespace" in comments:
                comments = re.sub(r'@namespace\s\w+', "", comments)
            if "@name" in comments:
                comments = re.sub(r'@name\s[\w.]+', "", comments)
            if "@function" in comments:
                comments = re.sub(r'@function', "", comments)
            if "@borrows" in comments:
                borrowsMatch = borrowsRe.search(comments)
                borrowsRoot = borrowsMatch.group(1)
                borrowsTarget = borrowsMatch.group(2)
                comments = re.sub(r'@borrows.*', "", comments)
                comments += "* "+borrowsTarget+" is shorthand for "+borrowsRoot
            if "@cfg" not in comments and "function" not in code and "class" not in code and code.strip():
                code = code.replace("async", "")
                funcMatch = functionCodeRe.search(code)
                if funcMatch:
                    comments += " * @method "+funcMatch.group(1)
                    code = ""
                else:
                    code = ""
                    # raise Exception("Can't find function: "+code+"\n\ncomments: "+comments)
            elif "@method" in comments:
                code = ""
            memberOfMatch = memberOfRe.search(comments)
            if memberOfMatch:
                memberOf = memberOfMatch.group(1)
            else:
                memberOf = ""
            if "class" in code:
                commentsList = re.compile(r'(\r\n|\r|\n)').split(comments)
                cleanCommentsList = [line for line in commentsList if not re.search(r'@(class|memberof)\b', line)]
                comments = ''.join(cleanCommentsList)
                comments = re.compile(r'(\r\n|\r|\n)(\r\n|\r|\n)+').sub("\n", comments)
                classMatch = classCodeRe.search(code)
                if classMatch:
                    comments += "* @class "+memberOf+"."+classMatch.group(1)
                    code = ""
                else:
                    raise Exception("Unable to find class")
            elif memberOf != "":
                comments = comments.replace("@memberof", "@member", 1)
            code = code.replace("...", "")
            code = defaultsRe.sub("", code)
            if "@cfg" in comments or "@class" in comments or "@method" in comments:
                output += "/**\n"+comments.strip()+"\n */\n"+code+"\n\n"
    return output    
    
def read_dir(dir):
    output = ""
    for root, subFolders, files in os.walk(dir):
        for f in files:
            if f.endswith(".js"):
                fullfilename = os.path.join(root, f)
                output += parse_file(fullfilename)
    return output

def main(argv):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dir', nargs='+', required=True, help='input directory (can have multiple values)')
    parser.add_argument('--output-file', required=True, help='output file')
    parser.add_argument('--exclude-file', nargs='+', help='exclude file')
    args = parser.parse_args()
    output = ""
    for input_dir in args.input_dir:
        for filename in os.listdir(input_dir):
            if filename.endswith(".js"):
                fullfilename = os.path.join(input_dir, filename)
                if not args.exclude_file or fullfilename not in args.exclude_file:
                    output += parse_file(fullfilename)
    with open(args.output_file, 'w') as f:
        f.write(output)

if __name__ == "__main__":
   main(sys.argv[1:])