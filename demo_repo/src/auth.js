export function loginUser(email, password) {
    if (!email) {
        throw new Error('missing email');
    }
    return AuthService.authenticate(email, password);
}

class AuthService {
    static authenticate(email, password) {
        return { email, password, token: createToken(email) };
    }
}

function createToken(email) {
    return `token:${email}`;
}
